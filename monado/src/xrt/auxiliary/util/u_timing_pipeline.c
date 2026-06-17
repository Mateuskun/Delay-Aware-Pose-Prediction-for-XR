// Copyright 2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Simplified timing pipeline CSVs, see @ref u_timing_pipeline.h.
 * @ingroup aux_util
 */

#include "util/u_timing_pipeline.h"

#include "util/u_frame_timing_writer.h"
#include "util/u_misc.h"

#include "os/os_threading.h"

#include <stdbool.h>
#include <stdlib.h>


/*
 *
 * Defines
 *
 */

#define DISPLAY_ROW_COUNT 256

enum display_col
{
	DISPLAY_TIME = 0,
	WAIT_FRAME,
	BEGIN_FRAME,
	LOCATE_VIEWS,
	PREDICT_FILTER,
	PRESENT,
	FRAME_ID,
	DISPLAY_COL_COUNT,
};

static const char *const display_columns[DISPLAY_COL_COUNT] = {
    [DISPLAY_TIME] = "display_time",
    [WAIT_FRAME] = "wait_frame",
    [BEGIN_FRAME] = "begin_frame",
    [LOCATE_VIEWS] = "locate_views",
    [PREDICT_FILTER] = "predict_filter",
    [PRESENT] = "present",
    [FRAME_ID] = "frame_id",
};

enum camera_col
{
	CAMERA_EXPOSURE = 0,
	CAMERA_USB_TRANSFER_DONE,
	CAMERA_SENT_TO_BASALT,
	CAMERA_FLUSHED,
	CAMERA_INDEX,
	CAMERA_SOURCE_SEQUENCE,
	CAMERA_COL_COUNT,
};

static const char *const camera_columns[CAMERA_COL_COUNT] = {
    [CAMERA_EXPOSURE] = "exposure",
    [CAMERA_USB_TRANSFER_DONE] = "usb_transfer_done",
    [CAMERA_SENT_TO_BASALT] = "sent_to_basalt",
    [CAMERA_FLUSHED] = "flushed",
    [CAMERA_INDEX] = "cam_index",
    [CAMERA_SOURCE_SEQUENCE] = "source_sequence",
};


/*
 *
 * Structs
 *
 */

struct display_row
{
	bool active;
	bool emitted;
	int64_t frame_id;
	int64_t display_time_ns;
	int64_t host_wait_frame_ns;
	int64_t host_begin_frame_ns;
	int64_t host_locate_views_ns;
	int64_t host_predict_filter_ns;
	int64_t host_present_ns;
};

struct timing_pipeline
{
	bool initialized;
	struct os_mutex mutex;
	struct u_frame_timing_writer *display_writer;
	struct u_frame_timing_writer *camera_writer;
	struct display_row display_rows[DISPLAY_ROW_COUNT];
};


/*
 *
 * State
 *
 */

static struct timing_pipeline g_tp = {0};
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;


/*
 *
 * Helpers
 *
 */

static void
initialize_once(void)
{
	if (g_tp.initialized) {
		return;
	}

	os_mutex_init(&g_tp.mutex);
	g_tp.initialized = true;

	const char *display_path = getenv("MONADO_DISPLAY_TIMING_CSV");
	if (display_path != NULL && display_path[0] != '\0') {
		g_tp.display_writer = u_frame_timing_writer_create(display_path, display_columns, DISPLAY_COL_COUNT);
	}

	const char *camera_path = getenv("MONADO_CAMERA_TIMING_CSV");
	if (camera_path != NULL && camera_path[0] != '\0') {
		g_tp.camera_writer = u_frame_timing_writer_create(camera_path, camera_columns, CAMERA_COL_COUNT);
	}
}

static void
ensure_initialized(void)
{
	if (g_tp.initialized) {
		return;
	}

	pthread_mutex_lock(&g_init_mutex);
	initialize_once();
	pthread_mutex_unlock(&g_init_mutex);
}

static int64_t
value_or_unset(int64_t ns)
{
	if (ns == 0 || ns == U_FTW_UNSET) {
		return U_FTW_UNSET;
	}
	return ns;
}

static struct display_row *
get_display_row_by_frame_id_locked(int64_t frame_id)
{
	if (frame_id < 0) {
		return NULL;
	}

	size_t index = (size_t)((uint64_t)frame_id % DISPLAY_ROW_COUNT);
	struct display_row *row = &g_tp.display_rows[index];
	if (row->active && row->frame_id == frame_id) {
		return row;
	}

	U_ZERO(row);
	row->active = true;
	row->frame_id = frame_id;
	return row;
}

static struct display_row *
get_display_row_by_display_time_locked(int64_t display_time_ns)
{
	for (size_t i = 0; i < DISPLAY_ROW_COUNT; i++) {
		struct display_row *row = &g_tp.display_rows[i];
		if (row->active && row->display_time_ns == display_time_ns) {
			return row;
		}
	}

	for (size_t i = 0; i < DISPLAY_ROW_COUNT; i++) {
		struct display_row *row = &g_tp.display_rows[i];
		if (!row->active) {
			U_ZERO(row);
			row->active = true;
			row->frame_id = U_FTW_UNSET;
			row->display_time_ns = display_time_ns;
			return row;
		}
	}

	return NULL;
}

static void
emit_display_row_locked(struct display_row *row)
{
	if (g_tp.display_writer == NULL || row == NULL || row->emitted) {
		return;
	}

	int64_t values[DISPLAY_COL_COUNT] = {0};
	values[DISPLAY_TIME] = value_or_unset(row->display_time_ns);
	values[WAIT_FRAME] = value_or_unset(row->host_wait_frame_ns);
	values[BEGIN_FRAME] = value_or_unset(row->host_begin_frame_ns);
	values[LOCATE_VIEWS] = value_or_unset(row->host_locate_views_ns);
	values[PREDICT_FILTER] = value_or_unset(row->host_predict_filter_ns);
	values[PRESENT] = value_or_unset(row->host_present_ns);
	values[FRAME_ID] = row->frame_id;

	u_frame_timing_writer_push(g_tp.display_writer, values);
	row->emitted = true;
}


/*
 *
 * Public functions
 *
 */

void
u_timing_pipeline_push_camera(int64_t exposure_ns,
                              int64_t host_usb_transfer_done_ns,
                              int64_t host_sent_to_basalt_ns,
                              int64_t host_flushed_ns,
                              int64_t cam_index,
                              int64_t source_sequence)
{
	ensure_initialized();

	if (g_tp.camera_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	int64_t values[CAMERA_COL_COUNT] = {0};
	values[CAMERA_EXPOSURE] = value_or_unset(exposure_ns);
	values[CAMERA_USB_TRANSFER_DONE] = value_or_unset(host_usb_transfer_done_ns);
	values[CAMERA_SENT_TO_BASALT] = value_or_unset(host_sent_to_basalt_ns);
	values[CAMERA_FLUSHED] = value_or_unset(host_flushed_ns);
	values[CAMERA_INDEX] = cam_index;
	values[CAMERA_SOURCE_SEQUENCE] = source_sequence;
	u_frame_timing_writer_push(g_tp.camera_writer, values);
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_mark_wait_frame(int64_t frame_id, int64_t display_time_ns, int64_t host_wait_frame_ns)
{
	ensure_initialized();
	if (g_tp.display_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	struct display_row *row = get_display_row_by_frame_id_locked(frame_id);
	if (row != NULL) {
		row->display_time_ns = display_time_ns;
		row->host_wait_frame_ns = host_wait_frame_ns;
	}
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_mark_begin_frame(int64_t frame_id, int64_t host_begin_frame_ns)
{
	ensure_initialized();
	if (g_tp.display_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	struct display_row *row = get_display_row_by_frame_id_locked(frame_id);
	if (row != NULL) {
		row->host_begin_frame_ns = host_begin_frame_ns;
	}
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_mark_locate_views(int64_t display_time_ns, int64_t host_locate_views_ns)
{
	ensure_initialized();
	if (g_tp.display_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	struct display_row *row = get_display_row_by_display_time_locked(display_time_ns);
	if (row != NULL) {
		row->host_locate_views_ns = host_locate_views_ns;
	}
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_mark_predict_filter(int64_t display_time_ns, int64_t host_predict_filter_ns)
{
	ensure_initialized();
	if (g_tp.display_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	struct display_row *row = get_display_row_by_display_time_locked(display_time_ns);
	if (row != NULL) {
		row->host_predict_filter_ns = host_predict_filter_ns;
	}
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_mark_present_frame(int64_t frame_id, int64_t host_present_ns)
{
	ensure_initialized();
	if (g_tp.display_writer == NULL) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	struct display_row *row = get_display_row_by_frame_id_locked(frame_id);
	if (row != NULL) {
		row->host_present_ns = host_present_ns;
		emit_display_row_locked(row);
	}
	os_mutex_unlock(&g_tp.mutex);
}

void
u_timing_pipeline_shutdown(void)
{
	if (!g_tp.initialized) {
		return;
	}

	os_mutex_lock(&g_tp.mutex);
	u_frame_timing_writer_destroy(&g_tp.display_writer);
	u_frame_timing_writer_destroy(&g_tp.camera_writer);
	os_mutex_unlock(&g_tp.mutex);
	os_mutex_destroy(&g_tp.mutex);
	U_ZERO(&g_tp);
}
