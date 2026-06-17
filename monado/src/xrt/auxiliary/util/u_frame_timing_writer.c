// Copyright 2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Generic per-frame CSV timing writer, see @ref u_frame_timing_writer.h.
 * @author Mateus
 * @ingroup aux_util
 */

#include "util/u_frame_timing_writer.h"
#include "util/u_logging.h"
#include "util/u_misc.h"

#include "os/os_threading.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct u_frame_timing_writer
{
	FILE *file;
	char **columns;
	size_t column_count;
	struct os_mutex mutex;
};

struct u_frame_timing_writer *
u_frame_timing_writer_create(const char *path, const char *const *column_names, size_t column_count)
{
	if (path == NULL || column_names == NULL || column_count == 0) {
		return NULL;
	}

	FILE *file = fopen(path, "w");
	if (file == NULL) {
		U_LOG_E("u_frame_timing_writer: could not open '%s' for writing", path);
		return NULL;
	}

	struct u_frame_timing_writer *ftw = U_TYPED_CALLOC(struct u_frame_timing_writer);
	ftw->file = file;
	ftw->column_count = column_count;
	ftw->columns = U_TYPED_ARRAY_CALLOC(char *, column_count);
	for (size_t i = 0; i < column_count; i++) {
		ftw->columns[i] = strdup(column_names[i]);
	}
	os_mutex_init(&ftw->mutex);

	fputc('#', ftw->file);
	for (size_t i = 0; i < column_count; i++) {
		fputs(ftw->columns[i], ftw->file);
		fputc(i + 1 == column_count ? '\n' : ',', ftw->file);
	}
	fflush(ftw->file);

	U_LOG_I("u_frame_timing_writer: writing to '%s' (%zu columns)", path, column_count);
	return ftw;
}

void
u_frame_timing_writer_push(struct u_frame_timing_writer *ftw, const int64_t *values)
{
	if (ftw == NULL || values == NULL) {
		return;
	}

	os_mutex_lock(&ftw->mutex);
	for (size_t i = 0; i < ftw->column_count; i++) {
		if (values[i] != U_FTW_UNSET) {
			fprintf(ftw->file, "%" PRId64, values[i]);
		}
		fputc(i + 1 == ftw->column_count ? '\n' : ',', ftw->file);
	}
	fflush(ftw->file);
	os_mutex_unlock(&ftw->mutex);
}

void
u_frame_timing_writer_flush(struct u_frame_timing_writer *ftw)
{
	if (ftw == NULL) {
		return;
	}
	os_mutex_lock(&ftw->mutex);
	fflush(ftw->file);
	os_mutex_unlock(&ftw->mutex);
}

void
u_frame_timing_writer_destroy(struct u_frame_timing_writer **ftw_ptr)
{
	if (ftw_ptr == NULL || *ftw_ptr == NULL) {
		return;
	}
	struct u_frame_timing_writer *ftw = *ftw_ptr;

	os_mutex_lock(&ftw->mutex);
	fflush(ftw->file);
	fclose(ftw->file);
	ftw->file = NULL;
	for (size_t i = 0; i < ftw->column_count; i++) {
		free(ftw->columns[i]);
	}
	free(ftw->columns);
	os_mutex_unlock(&ftw->mutex);
	os_mutex_destroy(&ftw->mutex);

	free(ftw);
	*ftw_ptr = NULL;
}
