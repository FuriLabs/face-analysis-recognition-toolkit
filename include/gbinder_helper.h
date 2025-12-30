/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#ifndef GBINDER_HELPER_H
#define GBINDER_HELPER_H

#include <gbinder.h>

/**
 * @needle: substring to search for (e.g. "android.hardware.neuralnetworks")
 *
 * Returns: TRUE if any listed service contains @needle as a substring,
 *          FALSE on not found or on any error.
 */
gboolean find_hal(const char *needle);

#endif // GBINDER_HELPER_H
