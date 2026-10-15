// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cmath>

#include "common/assert.h"
#include "common/debug.h"
#include "common/logging/log.h"
#include "common/perf_profiler.h"
#include "common/sampling_profiler.h"
#include "common/scope_exit.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

namespace {

/// Draws, dispatches, barriers, render passes, switches between draws and dispatches and barriers
/// right before them counted so far, what the GPU time of command buffers is fit to.
Scheduler::CostCounts CountCosts() {
    using Common::Perf::Counter;
    return {
        Common::Perf::Total(Counter::Draws),        Common::Perf::Total(Counter::Dispatches),
        Common::Perf::Total(Counter::Barriers),     Common::Perf::Total(Counter::RenderPasses),
        Common::Perf::Total(Counter::WorkSwitches), Common::Perf::Total(Counter::SwitchBarriers)};
}

} // Anonymous namespace

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance, bool record_on_thread, vk::Queue queue_)
    : instance{instance}, queue{queue_ ? queue_ : instance.GetGraphicsQueue()},
      on_graphics_queue{queue == instance.GetGraphicsQueue()},
      queue_mutex{on_graphics_queue ? submit_mutex : own_queue_mutex}, work_semaphore{instance},
      command_pool{instance, &work_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    if (record_on_thread) {
        stream = std::make_unique<CommandStream>();
    }
    BeginSession();
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
    if (stream) {
        recording_thread = std::jthread(std::bind_front(&Scheduler::RecordingThread, this));
    }
}

Scheduler::~Scheduler() {
    if (recording_thread.joinable()) {
        recording_thread.request_stop();
        stream->Wake();
        recording_thread.join();
    }
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::MeasureGpuTime() {
    const auto physical_device = instance.GetPhysicalDevice();
    const auto families = physical_device.getQueueFamilyProperties();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();
    const u32 valid_bits = family < families.size() ? families[family].timestampValidBits : 0;
    if (valid_bits == 0) {
        return;
    }
    const vk::QueryPoolCreateInfo pool_ci = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumTimestampPairs * 2,
    };
    auto [result, pool] = instance.GetDevice().createQueryPoolUnique(pool_ci);
    if (result != vk::Result::eSuccess) {
        return;
    }
    timestamp_pool = std::move(pool);
    const vk::QueryPoolCreateInfo run_pool_ci = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumTimestampPairs * RunMarksPerPair,
    };
    auto [run_result, new_run_pool] = instance.GetDevice().createQueryPoolUnique(run_pool_ci);
    if (run_result == vk::Result::eSuccess) {
        run_pool = std::move(new_run_pool);
    }
    const vk::QueryPoolCreateInfo dispatch_pool_ci = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = MaxTimedDispatches * 3,
    };
    auto [dispatch_result, new_dispatch_pool] =
        instance.GetDevice().createQueryPoolUnique(dispatch_pool_ci);
    if (dispatch_result == vk::Result::eSuccess) {
        dispatch_pool = std::move(new_dispatch_pool);
    }
    timestamp_period_ns = physical_device.getProperties().limits.timestampPeriod;
    timestamp_mask = valid_bits >= 64 ? ~u64{0} : (u64{1} << valid_bits) - 1;
}

void Scheduler::CollectGpuTimes() {
    while (!submitted_timestamps.empty() &&
           work_semaphore.IsFree(submitted_timestamps.front().tick)) {
        const SubmittedTimestamps submitted = std::move(submitted_timestamps.front());
        const u32 pair = submitted.pair;
        const auto& counts = submitted.counts;
        submitted_timestamps.pop_front();
        std::array<u64, 2> times{};
        const auto result = instance.GetDevice().getQueryPoolResults(
            *timestamp_pool, pair * 2, 2, sizeof(times), times.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        const u64 start = times[0] & timestamp_mask;
        const u64 end = times[1] & timestamp_mask;
        if (result != vk::Result::eSuccess || end <= start) {
            continue;
        }
        // Command buffers can overlap on the GPU, so only the time past the latest end counted
        // is added.
        const u64 counted_start = std::max(start, counted_gpu_end);
        if (end > counted_start) {
            Common::Perf::Count(
                Common::Perf::Counter::GpuBusyNs,
                static_cast<u64>(static_cast<double>(end - counted_start) * timestamp_period_ns));
            if (!submitted.run_marks_overflow) {
                CountWorkRuns(pair, counted_start, end, submitted.first_run_compute,
                              submitted.run_marks);
            }
        }
        counted_gpu_end = std::max(counted_gpu_end, end);
        if (counts) {
            FitGpuCost(static_cast<double>(end - start) * timestamp_period_ns / 1000.0, *counts);
        }
    }
    CollectDispatchTimes();
}

void Scheduler::CollectDispatchTimes() {
    if (!submitted_dispatches.empty() && work_semaphore.IsFree(submitted_dispatches_tick)) {
        // Each query comes with whether it was written: a dispatch whose marks weren't all
        // written is left out.
        const u32 num_queries = static_cast<u32>(submitted_dispatches.size()) * 3;
        std::vector<u64> results(num_queries * 2);
        const auto result = instance.GetDevice().getQueryPoolResults(
            *dispatch_pool, 0, num_queries, results.size() * sizeof(u64), results.data(),
            2 * sizeof(u64),
            vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
        if (result == vk::Result::eSuccess || result == vk::Result::eNotReady) {
            const auto mark_time = [&](size_t dispatch, u32 mark) -> std::optional<u64> {
                const size_t query = dispatch * 3 + mark;
                if (results[query * 2 + 1] == 0) {
                    return std::nullopt;
                }
                return results[query * 2] & timestamp_mask;
            };
            const auto to_ns = [&](u64 ticks) {
                return static_cast<u64>(static_cast<double>(ticks) * timestamp_period_ns);
            };
            for (size_t i = 0; i < submitted_dispatches.size(); ++i) {
                const auto& dispatch = submitted_dispatches[i];
                const auto begin = mark_time(i, 0);
                const auto start = mark_time(i, 1);
                const auto end = mark_time(i, 2);
                if (dispatch.marks != 3 || !begin || !start || !end || *start < *begin ||
                    *end < *start) {
                    continue;
                }
                const u64 dispatch_ns = to_ns(*end - *start);
                if (dispatch.is_draw) {
                    auto& by_pixel_shader = draw_times[dispatch.program_hash];
                    by_pixel_shader.draw_ns += dispatch_ns;
                    by_pixel_shader.setup_ns += to_ns(*start - *begin);
                    ++by_pixel_shader.count;
                    by_pixel_shader.vertices += dispatch.groups;
                    by_pixel_shader.vertex_hash = dispatch.vertex_hash;
                    continue;
                }
                const size_t kind = dispatch.after_draws ? 1 : 0;
                before_dispatch_ns[kind] += to_ns(*start - *begin);
                auto& by_ring = dispatch_times_by_ring[dispatch.compute_ring ? 1 : 0];
                by_ring.dispatch_ns += to_ns(*end - *begin);
                ++by_ring.count;
                by_ring.groups += dispatch.groups;
                auto& by_kind = dispatch_times_by_kind[kind];
                by_kind.dispatch_ns += dispatch_ns;
                ++by_kind.count;
                by_kind.groups += dispatch.groups;
                auto& by_program = dispatch_times[dispatch.program_hash];
                by_program.dispatch_ns += dispatch_ns;
                ++by_program.count;
                by_program.groups += dispatch.groups;
            }
            ++timed_command_buffers;
        }
        submitted_dispatches.clear();
    }

    const auto now = std::chrono::steady_clock::now();
    if (last_dispatch_report == std::chrono::steady_clock::time_point{}) {
        last_dispatch_report = now;
    }
    if (now - last_dispatch_report < std::chrono::seconds{10} || timed_command_buffers == 0) {
        return;
    }
    last_dispatch_report = now;

    // Which programs take the GPU's time, and whether going from draws to dispatches costs
    // more than the dispatches after it.
    const auto per_dispatch_us = [](u64 ns, u64 count) {
        return count == 0 ? 0.0 : static_cast<double>(ns) / 1000.0 / static_cast<double>(count);
    };
    std::vector<std::pair<u64, DispatchTimes>> programs(dispatch_times.begin(),
                                                        dispatch_times.end());
    const size_t num_top = std::min<size_t>(10, programs.size());
    std::partial_sort(
        programs.begin(), programs.begin() + num_top, programs.end(),
        [](const auto& a, const auto& b) { return a.second.dispatch_ns > b.second.dispatch_ns; });
    u64 total_ns{};
    for (const auto& [hash, times] : programs) {
        total_ns += times.dispatch_ns;
    }
    std::string top;
    for (size_t i = 0; i < num_top; ++i) {
        const auto& [hash, times] = programs[i];
        top += fmt::format(
            "{}cs_{:#018x} {:.0f}% ({} of {:.1f} us, {:.0f} groups)", i == 0 ? "" : ", ", hash,
            total_ns == 0
                ? 0.0
                : static_cast<double>(times.dispatch_ns) * 100.0 / static_cast<double>(total_ns),
            times.count, per_dispatch_us(times.dispatch_ns, times.count),
            static_cast<double>(times.groups) / static_cast<double>(std::max<u64>(times.count, 1)));
    }
    const auto& after_draws = dispatch_times_by_kind[1];
    const auto& after_dispatches = dispatch_times_by_kind[0];
    const auto& graphics_ring = dispatch_times_by_ring[0];
    const auto& compute_rings = dispatch_times_by_ring[1];
    const u64 ring_ns = graphics_ring.dispatch_ns + compute_rings.dispatch_ns;
    LOG_INFO(Render_Vulkan,
             "Dispatches timed one by one in {} command buffers: {:.1f} us for each of {} after "
             "draws, with {:.1f} us before it for barriers and the end of rendering, {:.1f} us for "
             "each of {} after dispatches, with {:.1f} us before it; {:.0f}% of the time with "
             "what came before them in the {} from the compute rings, {:.1f} us each, and the "
             "rest in the {} from the graphics ring, {:.1f} us each; {:.1f} ms in {} programs, "
             "the most in {}",
             timed_command_buffers, per_dispatch_us(after_draws.dispatch_ns, after_draws.count),
             after_draws.count, per_dispatch_us(before_dispatch_ns[1], after_draws.count),
             per_dispatch_us(after_dispatches.dispatch_ns, after_dispatches.count),
             after_dispatches.count, per_dispatch_us(before_dispatch_ns[0], after_dispatches.count),
             ring_ns == 0 ? 0.0
                          : static_cast<double>(compute_rings.dispatch_ns) * 100.0 /
                                static_cast<double>(ring_ns),
             compute_rings.count, per_dispatch_us(compute_rings.dispatch_ns, compute_rings.count),
             graphics_ring.count, per_dispatch_us(graphics_ring.dispatch_ns, graphics_ring.count),
             static_cast<double>(total_ns) / 1'000'000.0, programs.size(), top);
    ReportDrawTimes();
    dispatch_times.clear();
    dispatch_times_by_kind = {};
    dispatch_times_by_ring = {};
    before_dispatch_ns = {};
    timed_command_buffers = 0;
}

void Scheduler::ReportDrawTimes() {
    // Draws are timed one by one like dispatches, so each waits for all work before it and its
    // time is its own, a little more than it costs overlapped with other work. Which pixel
    // shaders take the most is what this is for.
    std::vector<std::pair<u64, DrawTimes>> programs(draw_times.begin(), draw_times.end());
    draw_times.clear();
    if (programs.empty()) {
        return;
    }
    const size_t num_top = std::min<size_t>(10, programs.size());
    std::partial_sort(
        programs.begin(), programs.begin() + num_top, programs.end(),
        [](const auto& a, const auto& b) { return a.second.draw_ns > b.second.draw_ns; });
    u64 total_ns{};
    u64 total_count{};
    u64 setup_ns{};
    for (const auto& [hash, times] : programs) {
        total_ns += times.draw_ns;
        total_count += times.count;
        setup_ns += times.setup_ns;
    }
    std::string top;
    for (size_t i = 0; i < num_top; ++i) {
        const auto& [hash, times] = programs[i];
        const double count = static_cast<double>(std::max<u64>(times.count, 1));
        top += fmt::format("{}ps_{:#018x}+vs_{:#018x} {:.0f}% ({} of {:.1f} us, {:.0f} vertices)",
                           i == 0 ? "" : ", ", hash, times.vertex_hash,
                           total_ns == 0 ? 0.0
                                         : static_cast<double>(times.draw_ns) * 100.0 /
                                               static_cast<double>(total_ns),
                           times.count, static_cast<double>(times.draw_ns) / 1000.0 / count,
                           static_cast<double>(times.vertices) / count);
    }
    LOG_INFO(
        Render_Vulkan,
        "Draws timed one by one: {:.1f} ms in {} draws of {} pixel shaders, {:.1f} us "
        "each, with {:.1f} us before each for the render pass and barriers; the most in {}",
        static_cast<double>(total_ns) / 1'000'000.0, total_count, programs.size(),
        static_cast<double>(total_ns) / 1000.0 / static_cast<double>(std::max<u64>(total_count, 1)),
        static_cast<double>(setup_ns) / 1000.0 / static_cast<double>(std::max<u64>(total_count, 1)),
        top);
}

void Scheduler::CountWorkRuns(u32 pair, u64 start, u64 end, bool first_run_compute,
                              const RunMarks& run_marks) {
    // A run lasts from where it began, or the command buffer did, to where the next one began, or
    // the command buffer ended. Marks are written once the work before them is done.
    std::array<u64, RunMarksPerPair> marks{};
    if (!run_marks.empty()) {
        const auto result = instance.GetDevice().getQueryPoolResults(
            *run_pool, pair * RunMarksPerPair, static_cast<u32>(run_marks.size()),
            run_marks.size() * sizeof(u64), marks.data(), sizeof(u64),
            vk::QueryResultFlagBits::e64);
        if (result != vk::Result::eSuccess) {
            return;
        }
    }
    u64 run_start = start;
    bool compute = first_run_compute;
    const auto count_run = [&](u64 run_end) {
        run_end = std::min(run_end, end);
        if (run_end > run_start) {
            // Stands for the command buffers whose runs weren't marked too.
            const auto ns = static_cast<u64>(static_cast<double>(run_end - run_start) *
                                             timestamp_period_ns * RunMarkInterval);
            Common::Perf::Count(compute ? Common::Perf::Counter::GpuDispatchRunNs
                                        : Common::Perf::Counter::GpuDrawRunNs,
                                ns);
            run_start = run_end;
        }
    };
    for (size_t i = 0; i < run_marks.size(); ++i) {
        count_run(marks[i] & timestamp_mask);
        compute = run_marks[i];
    }
    count_run(end);
}

void Scheduler::FitGpuCost(double gpu_us, const CostCounts& counts) {
    // The GPU is busy most of a frame, and what in a command buffer takes its time decides what
    // would shorten it.
    static constexpr size_t NumTerms = NumCostTerms;
    std::array<double, NumTerms> x{};
    for (size_t i = 0; i < counts.size(); ++i) {
        x[i] = static_cast<double>(counts[i]);
    }
    x[NumTerms - 1] = 1.0;
    for (size_t i = 0; i < NumTerms; ++i) {
        for (size_t j = 0; j < NumTerms; ++j) {
            cost_xx[i][j] += x[i] * x[j];
        }
        cost_xy[i] += x[i] * gpu_us;
    }
    cost_yy += gpu_us * gpu_us;
    ++cost_samples;

    const auto now = std::chrono::steady_clock::now();
    if (last_cost_report == std::chrono::steady_clock::time_point{}) {
        last_cost_report = now;
    }
    if (now - last_cost_report < std::chrono::seconds{10} || cost_samples < 100) {
        return;
    }
    last_cost_report = now;

    // Solves the normal equations by elimination. A little ridge keeps a term that didn't vary,
    // like dispatches in a scene without any, from leaving them singular.
    auto a = cost_xx;
    auto b = cost_xy;
    double max_diagonal = 0.0;
    for (size_t i = 0; i < NumTerms; ++i) {
        max_diagonal = std::max(max_diagonal, a[i][i]);
    }
    for (size_t i = 0; i < NumTerms; ++i) {
        a[i][i] += max_diagonal * 1e-9 + 1e-12;
    }
    for (size_t col = 0; col < NumTerms; ++col) {
        size_t pivot = col;
        for (size_t row = col + 1; row < NumTerms; ++row) {
            if (std::abs(a[row][col]) > std::abs(a[pivot][col])) {
                pivot = row;
            }
        }
        std::swap(a[col], a[pivot]);
        std::swap(b[col], b[pivot]);
        for (size_t row = col + 1; row < NumTerms; ++row) {
            const double factor = a[row][col] / a[col][col];
            for (size_t k = col; k < NumTerms; ++k) {
                a[row][k] -= factor * a[col][k];
            }
            b[row] -= factor * b[col];
        }
    }
    std::array<double, NumTerms> cost{};
    for (size_t i = NumTerms; i-- > 0;) {
        double sum = b[i];
        for (size_t k = i + 1; k < NumTerms; ++k) {
            sum -= a[i][k] * cost[k];
        }
        cost[i] = sum / a[i][i];
    }

    // How much of the GPU time each part accounts for, and how well the fit follows it.
    const double samples = static_cast<double>(cost_samples);
    const double total_us = cost_xy[NumTerms - 1];
    std::array<double, NumTerms> share{};
    double explained = 0.0;
    double fitted_squares = 0.0;
    for (size_t i = 0; i < NumTerms; ++i) {
        share[i] = total_us > 0.0 ? cost[i] * cost_xx[i][NumTerms - 1] * 100.0 / total_us : 0.0;
        explained += cost[i] * cost_xy[i];
        for (size_t k = 0; k < NumTerms; ++k) {
            fitted_squares += cost[i] * cost_xx[i][k] * cost[k];
        }
    }
    const double residual = cost_yy - 2.0 * explained + fitted_squares;
    const double variation = cost_yy - total_us * total_us / samples;
    const double fit = variation > 0.0 ? 100.0 * (1.0 - residual / variation) : 0.0;
    LOG_INFO(Render_Vulkan,
             "Host GPU time of command buffers: {:.2f} us a draw ({:.0f}%), {:.2f} us a dispatch "
             "({:.0f}%), {:.2f} us a barrier ({:.0f}%), {:.2f} us a render pass ({:.0f}%), {:.2f} "
             "us a switch between draws and dispatches ({:.0f}%), {:.2f} us more if a barrier came "
             "right before it ({:.0f}%) and {:.0f} us each ({:.0f}%), following {:.0f}% of its "
             "variation over {} command buffers of {:.0f} us on average",
             cost[0], share[0], cost[1], share[1], cost[2], share[2], cost[3], share[3], cost[4],
             share[4], cost[5], share[5], cost[6], share[6], fit, cost_samples, total_us / samples);

    cost_xx = {};
    cost_xy = {};
    cost_yy = 0.0;
    cost_samples = 0;
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    Common::Perf::Count(Common::Perf::Counter::RenderPasses);
    is_rendering = true;
    render_state = new_state;

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    CommandBuffer().beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    CommandBuffer().endRendering();
}

CommandRecorder Scheduler::UploadCommandBuffer() {
    ++commands_requested;
    auto& session = sessions.back();
    if (!session.has_upload) {
        session.has_upload = true;
        Run([this](RecordingContext& context) {
            const vk::CommandBufferBeginInfo begin_info = {
                .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
            };
            context.upload = command_pool.Commit();
            Check(context.upload.begin(begin_info));
            // Work submitted before may still read or write what is copied to.
            const vk::MemoryBarrier2 barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                .dstAccessMask =
                    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
            };
            context.upload.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
            });
        });
    }
    if (stream) {
        return CommandRecorder{*stream, CommandTarget::Upload};
    }
    return CommandRecorder{direct_context.upload};
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    if (work_semaphore.IsFree(tick)) {
        return;
    }
    Common::Perf::ScopedStall stall{Common::Perf::Stall::GpuWait};
    work_semaphore.Wait(tick);
}

void Scheduler::WaitSubmitted(u64 tick) {
    if (!stream) {
        return;
    }
    for (u64 submitted = submitted_tick.load(std::memory_order_acquire); submitted < tick;
         submitted = submitted_tick.load(std::memory_order_acquire)) {
        submitted_tick.wait(submitted, std::memory_order_acquire);
    }
}

void Scheduler::PopPendingOperations() {
    // Called for every draw, so neither the lock nor the driver is asked when nothing waits. An
    // operation deferred meanwhile is run by a later call.
    if (num_pending_ops.load(std::memory_order_acquire) == 0) {
        return;
    }
    std::unique_lock lk(pending_ops_mutex);
    if (pending_ops.empty()) {
        return;
    }
    work_semaphore.Refresh();
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
    }
    num_pending_ops.store(pending_ops.size(), std::memory_order_release);
}

void Scheduler::MarkWorkRun(bool compute) {
    if (compute == run_compute) {
        return;
    }
    run_compute = compute;
    auto& session = sessions.back();
    if (!session.mark_runs) {
        return;
    }
    if (session.run_marks.size() == session.run_marks.capacity()) {
        session.run_marks_overflow = true;
        return;
    }
    // Written once the work before it is done, which is where the run it ends is over.
    CommandBuffer().writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *run_pool,
                                    session.timestamp_pair * RunMarksPerPair +
                                        static_cast<u32>(session.run_marks.size()));
    session.run_marks.push_back(compute);
}

void Scheduler::MarkDispatch(DispatchMark mark, u64 program_hash, u32 groups, bool after_draws,
                             bool compute_ring, bool is_draw, u64 vertex_hash) {
    if (!sessions.back().time_dispatches) {
        return;
    }
    if (mark == DispatchMark::Begin) {
        if (timed_dispatches.size() == MaxTimedDispatches) {
            return;
        }
        timed_dispatches.push_back({
            .program_hash = program_hash,
            .groups = groups,
            .after_draws = after_draws,
            .compute_ring = compute_ring,
            .marks = 0,
            .is_draw = is_draw,
            .vertex_hash = vertex_hash,
        });
    } else if (timed_dispatches.empty() ||
               timed_dispatches.back().marks != static_cast<u32>(mark)) {
        // Its earlier marks weren't written.
        return;
    }
    auto& dispatch = timed_dispatches.back();
    const u32 query = static_cast<u32>(timed_dispatches.size() - 1) * 3 + static_cast<u32>(mark);
    CommandBuffer().writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *dispatch_pool,
                                    query);
    dispatch.marks = static_cast<u32>(mark) + 1;
}

void Scheduler::BeginSession() {
    EndSession();

    auto& session = sessions.emplace_back();
    ++session_id;

    // Pairs are used in turn, so the next one is free unless all are waiting for results.
    vk::QueryPool timestamps{};
    vk::QueryPool runs{};
    u32 pair = NoTimestamps;
    if (timestamp_pool && submitted_timestamps.size() + 1 < NumTimestampPairs) {
        pair = next_timestamp_pair;
        next_timestamp_pair = (next_timestamp_pair + 1) % NumTimestampPairs;
        timestamps = *timestamp_pool;
        runs = run_pool ? *run_pool : vk::QueryPool{};
        session.timestamp_pair = pair;
        session.counts = CountCosts();
        session.thread = std::this_thread::get_id();
        session.first_run_compute = run_compute;
        session.mark_runs = run_pool && session_id % RunMarkInterval == 0;
        if (!session.mark_runs) {
            runs = vk::QueryPool{};
        }
    }
    Run([this, timestamps, runs, pair](RecordingContext& context) {
        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        context.primary = command_pool.Commit();
        Check(context.primary.begin(begin_info));
        if (pair == NoTimestamps) {
            return;
        }
        context.primary.resetQueryPool(timestamps, pair * 2, 2);
        context.primary.writeTimestamp2(vk::PipelineStageFlagBits2::eTopOfPipe, timestamps,
                                        pair * 2);
        if (runs) {
            context.primary.resetQueryPool(runs, pair * RunMarksPerPair, RunMarksPerPair);
        }
    });

    // Timed one at a time, as their results are read once the command buffer is done.
    if (dispatch_pool && submitted_dispatches.empty() && session_id % TimedDispatchInterval == 0) {
        session.time_dispatches = true;
        timed_dispatches.clear();
        CommandBuffer().resetQueryPool(*dispatch_pool, 0, MaxTimedDispatches * 3);
    }

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::EndSession() {
    if (sessions.empty()) {
        return;
    }

    if (on_session) {
        on_session();
    }

    EndRendering();
    const auto& session = sessions.back();
    const bool has_upload = session.has_upload;
    const u32 pair = session.timestamp_pair;
    const vk::QueryPool timestamps = pair != NoTimestamps ? *timestamp_pool : vk::QueryPool{};
    Run([this, has_upload, pair, timestamps](RecordingContext& context) {
        if (has_upload) {
            // The work recorded after the uploads reads and writes what they copied to.
            const vk::MemoryBarrier2 barrier = {
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask =
                    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            };
            context.upload.pipelineBarrier2(vk::DependencyInfo{
                .memoryBarrierCount = 1,
                .pMemoryBarriers = &barrier,
            });
            Check(context.upload.end());
            recorded.push_back(context.upload);
            context.upload = vk::CommandBuffer{};
        }
        if (pair != NoTimestamps) {
            context.primary.writeTimestamp2(vk::PipelineStageFlagBits2::eBottomOfPipe, timestamps,
                                            pair * 2 + 1);
        }
        Check(context.primary.end());
        recorded.push_back(context.primary);
        context.primary = vk::CommandBuffer{};
    });
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    if (on_pre_submit) {
        on_pre_submit(info);
    }
    // The queue is used by other threads too. Commands recorded on the recording thread are
    // submitted there, and only binding sparse memory needs it here.
    std::unique_lock lk{queue_mutex};
    const u64 signal_value = work_semaphore.NextTick();
    work_since_submit = 0;
    Common::Perf::Count(Common::Perf::Counter::Submits);

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }
    if (stream) {
        lk.unlock();
    }

    EndSession();

    for (const auto& session : sessions) {
        if (session.time_dispatches) {
            submitted_dispatches = std::move(timed_dispatches);
            submitted_dispatches_tick = signal_value;
            timed_dispatches.clear();
        }
        if (session.timestamp_pair != NoTimestamps) {
            // Counts are only told apart for the command buffer when one thread recorded it all,
            // and one whose dispatches were timed took longer for it.
            std::optional<CostCounts> counts;
            if (session.thread == std::this_thread::get_id() && !session.time_dispatches) {
                counts = CountCosts();
                for (size_t i = 0; i < counts->size(); ++i) {
                    (*counts)[i] -= session.counts[i];
                }
            }
            submitted_timestamps.push_back({
                .tick = signal_value,
                .pair = session.timestamp_pair,
                .counts = counts,
                .first_run_compute = session.first_run_compute,
                .run_marks = session.run_marks,
                .run_marks_overflow = !session.mark_runs || session.run_marks_overflow,
            });
        }
    }
    sessions.clear();

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    // What other threads have to do for the command buffer before it runs is what was asked of
    // them by now.
    const u64 host_work = host_work_mark ? host_work_mark() : 0;
    if (stream) {
        Run([this, info, signal_value, host_work](RecordingContext&) {
            WaitHostWork(host_work);
            if (info.submitted_first) {
                info.submitted_first->WaitSubmitted(info.submitted_first_tick);
            }
            std::scoped_lock lock{queue_mutex};
            SubmitRecorded(info, signal_value);
        });
        stream->Publish(true);
    } else {
        WaitHostWork(host_work);
        if (info.submitted_first) {
            info.submitted_first->WaitSubmitted(info.submitted_first_tick);
        }
        SubmitRecorded(info, signal_value);
    }

    work_semaphore.Refresh();
    CollectGpuTimes();
    BeginSession();

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::WaitHostWork(u64 mark) {
    if (host_work_wait) {
        host_work_wait(u64{mark});
    }
}

void Scheduler::SubmitRecorded(const SubmitInfo& info, u64 signal_value) {
    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = info.wait_stages.data(),
        .commandBufferCount = static_cast<u32>(recorded.size()),
        .pCommandBuffers = recorded.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    if (on_graphics_queue) {
        ImGui::Core::TextureManager::Submit();
    }
    auto submit_result = queue.submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    recorded.clear();

    submitted_tick.store(signal_value, std::memory_order_release);
    if (stream) {
        submitted_tick.notify_all();
    }
}

void Scheduler::RecordingThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuCommandRecorder");
    // It does what the driver takes of the GPU thread's work, so where its time goes is logged
    // next to that thread's.
    Common::Perf::SampleCurrentThread("recorder");
    SCOPE_EXIT {
        Common::Perf::StopSamplingCurrentThread();
    };

    RecordingContext context{};
    while (!stoken.stop_requested()) {
        const auto start = std::chrono::steady_clock::now();
        if (stream->Replay(context)) {
            const auto elapsed = std::chrono::steady_clock::now() - start;
            Common::Perf::Record(
                Common::Perf::Stall::CommandRecording,
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        }
        if (!stream->HasWork()) {
            stream->WaitForWork(stoken);
        }
    }
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void DynamicState::BindComputePipelineOnceBuilt(const CommandRecorder& cmdbuf,
                                                const Pipeline& pipeline) {
    cmdbuf.bindPipelineOnceBuilt(
        vk::PipelineBindPoint::eCompute,
        [](const void* object) { return static_cast<const Pipeline*>(object)->Handle(); },
        &pipeline);
    // Its handle isn't known yet, so the next pipeline is bound whichever it is.
    compute_pipeline = vk::Pipeline{};
    dirty_state.compute_pipeline = false;
}

void DynamicState::Commit(const Instance& instance, const CommandRecorder& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
    }
    if (dirty_state.rasterizer_discard_enable) {
        dirty_state.rasterizer_discard_enable = false;
        cmdbuf.setRasterizerDiscardEnable(rasterizer_discard_enable);
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants.data());
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
    if (dirty_state.line_width) {
        dirty_state.line_width = false;
        cmdbuf.setLineWidth(line_width);
    }
    if (dirty_state.feedback_loop_enabled && instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        dirty_state.feedback_loop_enabled = false;
        cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_enabled
                                                      ? vk::ImageAspectFlagBits::eColor
                                                      : vk::ImageAspectFlagBits::eNone);
        // Pipelines don't take this state dynamically, and binding one after it sets it back to
        // theirs, as draws always did before binding only pipelines that changed.
        dirty_state.graphics_pipeline = true;
    }
}

} // namespace Vulkan
