/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#include <gtk/gtk.h>
#include <gst/gst.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fart.h"

GtkWidget *status_label = NULL;
static FaceAnalysisRecognition *face_handle = NULL;

static const char *
enrollment_state_to_string(EnrollmentState state)
{
    switch (state) {
        case ENROLLMENT_FAIL: return "ENROLLMENT_FAIL";
        case ENROLLMENT_IN_PROGRESS: return "ENROLLMENT_IN_PROGRESS";
        case ENROLLMENT_COMPLETE: return "ENROLLMENT_COMPLETE";
        case ENROLLMENT_SAVE_FAILED: return "ENROLLMENT_SAVE_FAILED";
        case ENROLLMENT_MULTIPLE_FACES: return "ENROLLMENT_MULTIPLE_FACES";
        case ENROLLMENT_NO_FACE: return "ENROLLMENT_NO_FACE";
        case ENROLLMENT_BAD_LIGHTING: return "ENROLLMENT_BAD_LIGHTING";
        default: return "UNKNOWN_ENROLLMENT_STATE";
    }
}

static const char *
recognition_state_to_string(RecognitionState state)
{
    switch (state) {
        case RECOGNITION_FAIL: return "RECOGNITION_FAIL";
        case RECOGNITION_NO_FACE: return "RECOGNITION_NO_FACE";
        case RECOGNITION_MULTIPLE_FACES: return "RECOGNITION_MULTIPLE_FACES";
        case RECOGNITION_NOT_ENROLLED: return "RECOGNITION_NOT_ENROLLED";
        case RECOGNITION_RECOGNIZED: return "RECOGNITION_RECOGNIZED";
        case RECOGNITION_NOT_RECOGNIZED: return "RECOGNITION_NOT_RECOGNIZED";
        default: return "UNKNOWN_RECOGNITION_STATE";
    }
}

static gboolean
update_label(gpointer data)
{
    char *msg = (char *)data;
    gtk_label_set_text(GTK_LABEL(status_label), msg);
    g_free(msg);
    return FALSE;
}

static GstFlowReturn
on_new_sample(GstElement *sink, gpointer user_data)
{
    (void) user_data;

    GstSample *sample = NULL;
    g_signal_emit_by_name(sink, "pull-sample", &sample);
    if (!sample)
        return GST_FLOW_ERROR;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    if (!caps) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstStructure *s = gst_caps_get_structure(caps, 0);
    int width = 0, height = 0;
    if (!gst_structure_get_int(s, "width", &width) ||
        !gst_structure_get_int(s, "height", &height)) {
        gst_sample_unref(sample);
        return GST_FLOW_ERROR;
    }

    GstMapInfo map;
    if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        int channels = 3;

        if (!fart_is_enrolled(face_handle)) {
            int progress = 0;
            EnrollmentState enroll_state =
                fart_enroll(face_handle, map.data, width, height, channels, &progress);

            const char *st = enrollment_state_to_string(enroll_state);

            if (enroll_state == ENROLLMENT_IN_PROGRESS) {
                g_print("Enrollment state: %s (%d%%)\n", st, progress);
                g_idle_add(update_label, g_strdup_printf("%s (%d%%)", st, progress));
            } else if (enroll_state == ENROLLMENT_COMPLETE) {
                g_print("Enrollment state: %s (100%%)\n", st);
                g_idle_add(update_label, g_strdup_printf("%s (100%%)", st));
            } else {
                g_print("Enrollment state: %s\n", st);
                g_idle_add(update_label, g_strdup(st));
            }
        } else {
            RecognitionState recog_state =
                fart_recognize(face_handle, map.data, width, height, channels);

            const char *st = recognition_state_to_string(recog_state);
            g_print("Recognition state: %s\n", st);
            g_idle_add(update_label, g_strdup(st));
        }

        gst_buffer_unmap(buffer, &map);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

static char *
make_faceauth_data_dir(void)
{
    const char *home = getenv("HOME");
    if (!home || !*home)
        return g_strdup(".");

    return g_strdup_printf("%s/.local/share/faceauth", home);
}

int
main(int argc, char *argv[])
{
    gtk_init(&argc, &argv);
    gst_init(&argc, &argv);

    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Face Detection, Enrollment & Recognition");
    gtk_window_set_default_size(GTK_WINDOW(window), 640, 480);

    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_container_add(GTK_CONTAINER(window), vbox);

    status_label = gtk_label_new("Initializing...");
    gtk_widget_set_margin_top(status_label, 5);

    const char *detection_model = "models/detect-class1.tflite";
    const char *recognition_model = "models/mobile_face_net.tflite";

    char *data_dir = make_faceauth_data_dir();
    g_print("Using face data dir: %s\n", data_dir);

    face_handle = fart_create(detection_model, recognition_model, data_dir);
    g_free(data_dir);

    if (!face_handle) {
        fprintf(stderr, "Failed to create face detector\n");
        return EXIT_FAILURE;
    }

    const char *pipeline_str =
        "droidcamsrc camera_device=1 mode=2 ! tee name=t "
        "t. ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, width=640, height=480 ! videoconvert ! "
        "videoflip video-direction=auto ! gtksink name=sink sync=false "
        "t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! video/x-raw, format=BGR ! "
        "videoflip video-direction=auto ! appsink name=appsink max-buffers=1 drop=true emit-signals=true sync=false";

    GError *error = NULL;
    GstElement *pipeline = gst_parse_launch(pipeline_str, &error);
    if (error) {
        fprintf(stderr, "Error creating pipeline: %s\n", error->message);
        g_error_free(error);
        fart_destroy(face_handle);
        return EXIT_FAILURE;
    }

    GstElement *gtksink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    if (!gtksink) {
        fprintf(stderr, "Failed to get gtksink element from pipeline\n");
        gst_object_unref(pipeline);
        fart_destroy(face_handle);
        return EXIT_FAILURE;
    }

    GtkWidget *video_widget = NULL;
    g_object_get(G_OBJECT(gtksink), "widget", &video_widget, NULL);
    if (!video_widget) {
        fprintf(stderr, "Failed to retrieve video widget from gtksink\n");
        gst_object_unref(gtksink);
        gst_object_unref(pipeline);
        fart_destroy(face_handle);
        return EXIT_FAILURE;
    }

    gtk_box_pack_start(GTK_BOX(vbox), video_widget, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), status_label, FALSE, FALSE, 0);

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "appsink");
    if (!appsink) {
        fprintf(stderr, "Failed to get appsink element from pipeline\n");
        gst_object_unref(gtksink);
        gst_object_unref(pipeline);
        fart_destroy(face_handle);
        return EXIT_FAILURE;
    }

    g_object_set(G_OBJECT(appsink), "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), NULL);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    gtk_widget_show_all(window);

    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    gtk_main();

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(appsink);
    gst_object_unref(gtksink);
    gst_object_unref(pipeline);

    fart_destroy(face_handle);
    return EXIT_SUCCESS;
}
