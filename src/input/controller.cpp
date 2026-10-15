// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>

#include <SDL3/SDL.h>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "core/libraries/system/userservice.h"
#include "core/user_settings.h"
#include "input/controller.h"

namespace Input {

using Libraries::Pad::OrbisPadButtonDataOffset;

namespace {

void CalculateOrientation(const Libraries::Pad::OrbisFVector3& angular_velocity, float delta_time,
                          const Libraries::Pad::OrbisFQuaternion& last_orientation,
                          Libraries::Pad::OrbisFQuaternion& orientation) {
    if (delta_time > 1.0f) {
        orientation = last_orientation;
        return;
    }
    Libraries::Pad::OrbisFQuaternion q = last_orientation;
    const Libraries::Pad::OrbisFQuaternion omega = {angular_velocity.x, angular_velocity.y,
                                                    angular_velocity.z, 0.0f};

    const Libraries::Pad::OrbisFQuaternion q_omega = {
        q.w * omega.x + q.x * omega.w + q.y * omega.z - q.z * omega.y,
        q.w * omega.y + q.y * omega.w + q.z * omega.x - q.x * omega.z,
        q.w * omega.z + q.z * omega.w + q.x * omega.y - q.y * omega.x,
        q.w * omega.w - q.x * omega.x - q.y * omega.y - q.z * omega.z};

    const Libraries::Pad::OrbisFQuaternion q_dot = {0.5f * q_omega.x, 0.5f * q_omega.y,
                                                    0.5f * q_omega.z, 0.5f * q_omega.w};

    q.x += q_dot.x * delta_time;
    q.y += q_dot.y * delta_time;
    q.z += q_dot.z * delta_time;
    q.w += q_dot.w * delta_time;

    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    q.x /= norm;
    q.y /= norm;
    q.z /= norm;
    q.w /= norm;
    orientation = q;
}

using Quaternion = Libraries::Pad::OrbisFQuaternion;

constexpr float Gravity = 9.81f;
constexpr float HalfPi = 1.5707964f;
constexpr float TwoPi = 6.2831855f;
// A quarter turn in a quarter of a second, roughly how fast a hand turns a controller over.
constexpr float EmulatedTiltSpeed = TwoPi;
constexpr float EmulatedShakeFrequency = 5.0f;
constexpr float EmulatedShakeAmplitude = 2.0f * Gravity;

Quaternion Multiply(const Quaternion& a, const Quaternion& b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

Quaternion Conjugate(const Quaternion& q) {
    return {-q.x, -q.y, -q.z, q.w};
}

Quaternion Normalize(const Quaternion& q) {
    const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    return {q.x / norm, q.y / norm, q.z / norm, q.w / norm};
}

// Rotation by |rate| * dt radians around the direction of rate.
Quaternion FromRotationVector(const float rate[3], float dt) {
    const float speed = std::sqrt(rate[0] * rate[0] + rate[1] * rate[1] + rate[2] * rate[2]);
    if (speed * dt < 1e-9f) {
        return {0.0f, 0.0f, 0.0f, 1.0f};
    }
    const float s = std::sin(0.5f * speed * dt) / speed;
    return {rate[0] * s, rate[1] * s, rate[2] * s, std::cos(0.5f * speed * dt)};
}

} // namespace

GameController::GameController() : m_states_queue(64) {}

State GameController::ReadState() {
    std::lock_guard lock{m_state_mutex};
    return m_state;
}

int GameController::ReadStates(State* states, int states_num) {
    std::lock_guard lock{m_state_mutex};
    if (states_num <= 0) {
        return 0;
    }

    if (!m_state.connected) {
        states[0] = m_state;
        return 1;
    }

    if (states_num == 1) {
        // Retained history can make a later multi-sample read return up to 64 stale reports, so
        // mixed single- and multi-sample reads require dedicated tests.
        states[0] = m_state;
        return 1;
    }

    int read_count = 0;
    while (read_count < states_num) {
        auto state = m_states_queue.Pop();
        if (!state) {
            break;
        }
        states[read_count++] = std::move(*state);
    }
    return read_count;
}

void GameController::Button(OrbisPadButtonDataOffset button, bool is_pressed) {
    std::lock_guard lock{m_state_mutex};
    m_state.OnButton(button, is_pressed);
    PushStateLocked();
}

void GameController::Axis(Input::Axis axis, int value, bool smooth) {
    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    m_state.OnAxis(axis, value, timestamp, smooth);
    PushStateLocked(timestamp);
}

void GameController::UpdateGyro(const float gyro[3]) {
    std::lock_guard lock{m_state_mutex};
    m_motion_emulated = false;
    std::memcpy(gyro_buf, gyro, sizeof(gyro_buf));
}

void GameController::UpdateAcceleration(const float acceleration[3]) {
    std::lock_guard lock{m_state_mutex};
    m_motion_emulated = false;
    std::memcpy(accel_buf, acceleration, sizeof(accel_buf));
}

void GameController::SetEmulatedRotationRate(const float world_rate[3]) {
    std::lock_guard lock{m_state_mutex};
    EnableMotionEmulationLocked();
    std::memcpy(m_emu_world_rate, world_rate, sizeof(m_emu_world_rate));
}

void GameController::SetEmulatedTilt(TiltDirection direction, bool held) {
    std::lock_guard lock{m_state_mutex};
    EnableMotionEmulationLocked();
    (direction == TiltDirection::Left ? m_emu_tilt_left : m_emu_tilt_right) = held;
}

void GameController::SetEmulatedShake(bool shaking) {
    std::lock_guard lock{m_state_mutex};
    EnableMotionEmulationLocked();
    m_emu_shake = shaking;
}

void GameController::PollState() {
    std::lock_guard lock{m_state_mutex};
    PushStateLocked();
}

void GameController::ResetOrientation() {
    std::lock_guard lock{m_state_mutex};
    m_state.orientation = {0.0f, 0.0f, 0.0f, 1.0f};
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::SetTouchpadState(int touch_index, bool touch_down, float x, float y) {
    if (touch_index < 0 || touch_index >= 2) {
        return;
    }

    std::lock_guard lock{m_state_mutex};
    const u64 timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    const bool was_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    auto& touch = m_state.touchpad[touch_index];
    if (touch_down && !touch.state) {
        touch.ID = m_next_touch_id;
        m_next_touch_id = m_next_touch_id == 127 ? 1 : m_next_touch_id + 1;
    }
    m_state.OnTouchpad(touch_index, touch_down, x, y);
    const bool is_pressed = m_state.touchpad[0].state || m_state.touchpad[1].state;
    if (!was_pressed && is_pressed) {
        m_touch_down_timestamp = timestamp;
    } else if (was_pressed && !is_pressed) {
        m_touch_down_timestamp = 0;
    }
    PushStateLocked(timestamp);
}

void GameController::SwipeTouchpad(float from_x, float from_y, float to_x, float to_y,
                                   int fingers) {
    // Games read the touchpad every frame, so the finger has to move for a few of them to see
    // a swipe rather than a tap: about 160 ms in steps short enough to land in each frame.
    static constexpr int Steps = 16;
    static constexpr auto StepTime = std::chrono::milliseconds{10};
    fingers = std::clamp(fingers, 1, 2);
    // Two fingers sit either side of the line the swipe follows, across its direction.
    const bool horizontal = std::abs(to_x - from_x) >= std::abs(to_y - from_y);
    const float spread = fingers == 2 ? 0.1f : 0.0f;
    const float offset_x = horizontal ? 0.0f : spread;
    const float offset_y = horizontal ? spread : 0.0f;
    std::lock_guard lock{m_swipe_mutex};
    // Replacing the thread asks the one going on to stop, which lifts its fingers, and waits.
    m_swipe_thread = std::jthread([=, this](std::stop_token token) {
        const auto place = [&](bool down, float x, float y) {
            SetTouchpadState(0, down, x - offset_x, y - offset_y);
            if (fingers == 2) {
                SetTouchpadState(1, down, x + offset_x, y + offset_y);
            }
        };
        place(true, from_x, from_y);
        for (int step = 1; step <= Steps && !token.stop_requested(); ++step) {
            std::this_thread::sleep_for(StepTime);
            const float t = static_cast<float>(step) / Steps;
            place(true, from_x + (to_x - from_x) * t, from_y + (to_y - from_y) * t);
        }
        place(false, to_x, to_y);
    });
}

void GameController::ConnectController(SDL_Gamepad* pad) {
    std::lock_guard lock{m_state_mutex};
    m_sdl_gamepad = pad;
    if (override_colour) {
        SetLightBarRGB({});
    }
    m_states_queue.Clear();
    if (!m_state.connected) {
        ++m_state.connected_count;
        if (m_state.connected_count == 0) {
            m_state.connected_count = 1;
        }
    }
    m_state.connected = true;
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::DisconnectController() {
    std::lock_guard lock{m_state_mutex};
    m_states_queue.Clear();
    m_sdl_gamepad = nullptr;

    const u8 connected_count = m_state.connected_count;
    m_state = {};
    m_state.connected_count = connected_count;
    std::fill(gyro_buf, gyro_buf + 3, 0.0f);
    std::fill(accel_buf, accel_buf + 3, 0.0f);
    accel_buf[1] = 9.81f;
    m_motion_emulated = false;
    m_next_touch_id = 1;
    m_touch_down_timestamp = 0;
    m_state.connected = false;
    m_last_orientation_update = 0;
    PushStateLocked();
}

void GameController::UpdateOrientationLocked(u64 timestamp) {
    if (m_last_orientation_update == 0 || timestamp <= m_last_orientation_update) {
        m_last_orientation_update = timestamp;
        return;
    }
    const float delta_time =
        static_cast<float>(timestamp - m_last_orientation_update) / 1'000'000.f;
    Libraries::Pad::OrbisFQuaternion orientation{};
    CalculateOrientation(m_state.angularVelocity, delta_time, m_state.orientation, orientation);
    m_state.orientation = orientation;
    m_last_orientation_update = timestamp;
}

void GameController::EnableMotionEmulationLocked() {
    if (m_motion_emulated) {
        return;
    }
    m_motion_emulated = true;
    m_emu_aim = {0.0f, 0.0f, 0.0f, 1.0f};
    m_emu_pose = {0.0f, 0.0f, 0.0f, 1.0f};
    std::fill(m_emu_world_rate, m_emu_world_rate + 3, 0.0f);
    m_emu_roll = 0.0f;
    m_emu_tilt_left = false;
    m_emu_tilt_right = false;
    m_emu_shake = false;
    m_emu_shake_phase = 0.0f;
    m_last_motion_update = 0;
}

void GameController::UpdateEmulatedMotionLocked(u64 timestamp) {
    if (!m_motion_emulated) {
        return;
    }
    float dt = 0.0f;
    if (m_last_motion_update != 0 && timestamp > m_last_motion_update) {
        dt = std::min(static_cast<float>(timestamp - m_last_motion_update) / 1'000'000.f, 0.1f);
    }
    m_last_motion_update = timestamp;

    // Mouse rotation turns the controller around the world axes, so it aims the same way
    // whichever way the controller is tilted.
    m_emu_aim = Normalize(Multiply(FromRotationVector(m_emu_world_rate, dt), m_emu_aim));

    // Tilting turns the controller over around its own Z axis, left side or right side down.
    const float target_roll =
        (m_emu_tilt_left ? HalfPi : 0.0f) - (m_emu_tilt_right ? HalfPi : 0.0f);
    const float max_step = EmulatedTiltSpeed * dt;
    m_emu_roll += std::clamp(target_roll - m_emu_roll, -max_step, max_step);
    const Quaternion roll{0.0f, 0.0f, std::sin(0.5f * m_emu_roll), std::cos(0.5f * m_emu_roll)};
    const Quaternion pose = Multiply(m_emu_aim, roll);

    // The gyro reports the rotation between the previous pose and this one, in controller axes.
    Quaternion delta = Multiply(Conjugate(m_emu_pose), pose);
    if (delta.w < 0.0f) {
        delta = {-delta.x, -delta.y, -delta.z, -delta.w};
    }
    const float sin_half = std::sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z);
    if (dt > 0.0f && sin_half > 1e-6f) {
        const float scale = 2.0f * std::atan2(sin_half, delta.w) / (sin_half * dt);
        gyro_buf[0] = delta.x * scale;
        gyro_buf[1] = delta.y * scale;
        gyro_buf[2] = delta.z * scale;
    } else {
        std::fill(gyro_buf, gyro_buf + 3, 0.0f);
    }
    m_emu_pose = pose;

    // The accelerometer reports the reaction to gravity, plus the up and down motion of a shake,
    // turned into controller axes.
    float up = Gravity;
    if (m_emu_shake) {
        m_emu_shake_phase =
            std::fmod(m_emu_shake_phase + TwoPi * EmulatedShakeFrequency * dt, TwoPi);
        up += EmulatedShakeAmplitude * std::sin(m_emu_shake_phase);
    }
    const auto& q = pose;
    accel_buf[0] = up * 2.0f * (q.x * q.y + q.w * q.z);
    accel_buf[1] = up * (1.0f - 2.0f * (q.x * q.x + q.z * q.z));
    accel_buf[2] = up * 2.0f * (q.y * q.z - q.w * q.x);
}

void GameController::PushStateLocked(u64 timestamp) {
    if (timestamp == 0) {
        timestamp = Libraries::Kernel::sceKernelGetProcessTime();
    }
    m_state.UpdateAxisSmoothing(timestamp);
    UpdateEmulatedMotionLocked(timestamp);
    m_state.OnGyro(gyro_buf);
    m_state.OnAccel(accel_buf);
    UpdateOrientationLocked(timestamp);
    m_state.time = timestamp;
    m_state.touch_time_since_held_down =
        m_touch_down_timestamp == 0 ? 0 : timestamp - m_touch_down_timestamp;
    m_states_queue.Push(m_state);
}

void GameController::SetLightBarRGB(u8 const r, u8 const g, u8 const b) {
    if (override_colour.has_value()) {
        if (m_sdl_gamepad) {
            SDL_SetGamepadLED(m_sdl_gamepad, override_colour->r, override_colour->g,
                              override_colour->b);
        }
        return;
    }
    colour = {r, g, b};
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, r, g, b);
    }
}

void GameController::SetLightBarRGB(Colour const c) {
    SetLightBarRGB(c.r, c.g, c.b);
}

Colour GameController::GetLightBarRGB() {
    return colour;
}

void GameController::PollLightColour() {
    if (m_sdl_gamepad != nullptr) {
        SDL_SetGamepadLED(m_sdl_gamepad, colour.r, colour.g, colour.b);
    }
}

void GameControllers::ResetLightbarColors() {
    for (auto& c : controllers) {
        auto const* u = UserManagement.GetUserByID(c->user_id);
        if (!u || !c->m_sdl_gamepad) {
            continue;
        }
        auto const i = u->user_color - 1;
        if (i < 0 || i > 3) {
            continue;
        }
        auto const& col = g_user_colours[i];
        c->override_colour = std::nullopt;
        c->SetLightBarRGB(col);
    }
}

bool GameController::SetVibration(u8 smallMotor, u8 largeMotor) {
    if (m_sdl_gamepad != nullptr) {
        return SDL_RumbleGamepad(m_sdl_gamepad, (smallMotor / 255.0f) * 0xFFFF,
                                 (largeMotor / 255.0f) * 0xFFFF, -1);
    }
    return true;
}

static bool is_first_check = true;

void GameControllers::TryOpenSDLControllers() {
    using namespace Libraries::UserService;
    int controller_count;
    SDL_JoystickID* new_joysticks = SDL_GetGamepads(&controller_count);
    LOG_INFO(Input, "{} controllers are currently connected", controller_count);

    std::unordered_set<SDL_JoystickID> assigned_ids;
    std::array<bool, 4> slot_taken{false, false, false, false};

    for (int i = 0; i < 4; i++) {
        SDL_Gamepad* pad = controllers[i]->m_sdl_gamepad;
        if (pad) {
            SDL_JoystickID id = SDL_GetGamepadID(pad);
            bool still_connected = false;
            for (int j = 0; j < controller_count; j++) {
                if (new_joysticks[j] == id) {
                    still_connected = true;
                    assigned_ids.insert(id);
                    slot_taken[i] = true;
                    break;
                }
            }
            if (!still_connected) {
                SDL_CloseGamepad(pad);
                controllers[i]->DisconnectController();
                controllers[i]->user_id = -1;
                slot_taken[i] = false;
            }
        }
    }

    for (int j = 0; j < controller_count; j++) {
        SDL_JoystickID id = new_joysticks[j];
        if (assigned_ids.contains(id))
            continue;

        SDL_Gamepad* pad = SDL_OpenGamepad(id);
        if (!pad) {
            continue;
        }

        for (int i = 0; i < 4; i++) {
            if (!slot_taken[i]) {
                auto u = UserManagement.GetUserByPlayerIndex(i + 1);
                if (!u) {
                    LOG_INFO(Input, "User {} not found", i + 1);
                    continue; // for now, if you don't specify who Player N is in the config,
                              // Player N won't be registered at all
                }
                auto* c = controllers[i];
                LOG_INFO(Input, "Gamepad registered for slot {}! Handle: {}", i,
                         SDL_GetGamepadID(pad));
                slot_taken[i] = true;
                c->user_id = u->user_id;
                UserManagement.LoginUser(u, i + 1);
                c->ConnectController(pad);
                if (EmulatorSettings.IsMotionControlsEnabled()) {
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_GYRO, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_GYRO);
                        LOG_INFO(Input, "Gyro initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize gyro controls for gamepad {}",
                                  c->user_id);
                    }
                    if (SDL_SetGamepadSensorEnabled(c->m_sdl_gamepad, SDL_SENSOR_ACCEL, true)) {
                        const float poll_rate =
                            SDL_GetGamepadSensorDataRate(c->m_sdl_gamepad, SDL_SENSOR_ACCEL);
                        LOG_INFO(Input, "Accel initialized, poll rate: {}", poll_rate);
                    } else {
                        LOG_ERROR(Input, "Failed to initialize accel controls for gamepad {}",
                                  c->user_id);
                    }
                }
                break;
            }
        }
    }
    if (is_first_check) [[unlikely]] {
        is_first_check = false;
        if (controller_count == 0) {
            auto u = UserManagement.GetUserByPlayerIndex(1);
            controllers[0]->user_id = u->user_id;
            controllers[0]->ConnectController(nullptr);
            UserManagement.LoginUser(u, 1);
        }
    }
    SDL_free(new_joysticks);
}
u8 GameControllers::GetGamepadIndexFromJoystickId(SDL_JoystickID id) {
    auto g = SDL_GetGamepadFromID(id);
    ASSERT(g != nullptr);
    for (int i = 0; i < 5; i++) {
        if (controllers[i]->m_sdl_gamepad == g) {
            return i;
        }
    }
    // LOG_TRACE(Input, "Gamepad index: {}", index);
    return -1;
}

std::optional<u8> GameControllers::GetControllerIndexFromUserID(s32 user_id) {
    auto const u = UserManagement.GetUserByID(user_id);
    if (!u) {
        return std::nullopt;
    }
    return u->player_index - 1;
}

std::optional<u8> GameControllers::GetControllerIndexFromControllerID(s32 controller_id) {
    if (controller_id < 1 || controller_id > 5) {
        return std::nullopt;
    }
    return controller_id - 1;
}

} // namespace Input
