/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Bardia Moshiri <bardia@furilabs.com>
 */

#include "gbinder_helper.h"

gboolean
find_hal(const char *needle)
{
    GBinderServiceManager *sm = NULL;
    char **services = NULL;
    gboolean found = FALSE;

    if (!needle || !needle[0])
        return FALSE;

    sm = gbinder_servicemanager_new(GBINDER_DEFAULT_HWBINDER);
    if (!sm)
        return FALSE;

    if (!gbinder_servicemanager_wait(sm, 1000)) {
        gbinder_servicemanager_unref(sm);
        return FALSE;
    }

    services = gbinder_servicemanager_list_sync(sm);
    gbinder_servicemanager_unref(sm);

    if (!services)
        return FALSE;

    for (char **p = services; *p; ++p) {
        /* Partial match:
         * e.g. "android.hardware.neuralnetworks" matches
         * "android.hardware.neuralnetworks@1.3::IDevice/mtk-neuron"
         */
        if (g_strstr_len(*p, -1, needle) != NULL) {
            found = TRUE;
            break;
        }
    }

    g_strfreev(services);
    return found;
}
