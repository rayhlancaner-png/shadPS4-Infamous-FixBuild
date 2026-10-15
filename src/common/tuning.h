// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include "common/logging/log.h"
#include "common/types.h"

namespace Common {

/// Value of an environment variable that overrides a performance constant, to try values without
/// rebuilding. Returns default_value when it is unset or isn't a number. Logged once per call
/// site, so call it from a function-local static.
inline u64 Tune(const char* name, u64 default_value) {
    u64 value = default_value;
    if (const char* text = std::getenv(name)) {
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(text, &end, 10);
        if (end != text && *end == '\0') {
            value = parsed;
        }
    }
    LOG_INFO(Render, "Tuning: {} = {}{}", name, value, value == default_value ? "" : " (override)");
    return value;
}

} // namespace Common
