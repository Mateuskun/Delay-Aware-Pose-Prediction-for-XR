// Copyright 2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Timing pipeline CSVs (display.csv + camera.csv).
 * @ingroup aux_util
 */
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void
u_timing_pipeline_push_camera(int64_t exposure_ns,
                              int64_t host_usb_transfer_done_ns,
                              int64_t host_sent_to_basalt_ns,
                              int64_t host_flushed_ns,
                              int64_t cam_index,
                              int64_t source_sequence);

void
u_timing_pipeline_mark_wait_frame(int64_t frame_id, int64_t display_time_ns, int64_t host_wait_frame_ns);

void
u_timing_pipeline_mark_begin_frame(int64_t frame_id, int64_t host_begin_frame_ns);

void
u_timing_pipeline_mark_locate_views(int64_t display_time_ns, int64_t host_locate_views_ns);

void
u_timing_pipeline_mark_predict_filter(int64_t display_time_ns, int64_t host_predict_filter_ns);

void
u_timing_pipeline_mark_present_frame(int64_t frame_id, int64_t host_present_ns);

void
u_timing_pipeline_shutdown(void);

#ifdef __cplusplus
}
#endif
