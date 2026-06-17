// Copyright 2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Generic, thread-safe, extensible CSV writer for per-frame timing data.
 * @ingroup aux_util
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define U_FTW_UNSET INT64_MIN

struct u_frame_timing_writer;

struct u_frame_timing_writer *
u_frame_timing_writer_create(const char *path, const char *const *column_names, size_t column_count);

void
u_frame_timing_writer_push(struct u_frame_timing_writer *ftw, const int64_t *values);

void
u_frame_timing_writer_flush(struct u_frame_timing_writer *ftw);

void
u_frame_timing_writer_destroy(struct u_frame_timing_writer **ftw_ptr);

#ifdef __cplusplus
}
#endif
