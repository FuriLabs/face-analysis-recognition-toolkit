/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <glib.h>
#include <opencv2/opencv.hpp>

#include "fart.h"
#include "detector.h"

struct FaceAnalysisRecognition {
    FaceDetector *instance = nullptr;
};

extern "C" {

FaceAnalysisRecognition *
fart_create(const char *detection_model,
            const char *recognition_model,
            const char *anti_spoof_model,
            const char *data_dir,
            const char *enrollment_json)
{
    if (!detection_model) {
        g_debug("fart_create: detection_model is NULL");
        return nullptr;
    }
    if (!recognition_model) {
        g_debug("fart_create: recognition_model is NULL");
        return nullptr;
    }

    gboolean has_data_dir = data_dir && data_dir[0] != '\0';
    gboolean has_json = enrollment_json != nullptr;

    if (has_data_dir == has_json) {
        g_debug("fart_create: exactly one of data_dir or enrollment_json must be provided");
        return nullptr;
    }

    std::string anti_spoof_model_path;
    if (anti_spoof_model && anti_spoof_model[0] != '\0')
        anti_spoof_model_path = std::string(anti_spoof_model);

    FaceAnalysisRecognition *handle = new FaceAnalysisRecognition();
    if (!handle) {
        g_debug("fart_create: failed to allocate handle");
        return nullptr;
    }

    try {
        handle->instance = new FaceDetector(std::string(detection_model),
                                            std::string(recognition_model),
                                            anti_spoof_model_path,
                                            has_data_dir ? data_dir : nullptr,
                                            has_json ? enrollment_json : nullptr);
    } catch (const std::exception &e) {
        std::cerr << "Error in fart_create: " << e.what() << std::endl;
        g_debug("fart_create: exception creating FaceDetector: %s", e.what());
        delete handle;
        return nullptr;
    } catch (...) {
        g_debug("fart_create: unknown exception creating FaceDetector");
        delete handle;
        return nullptr;
    }

    if (!handle->instance) {
        g_debug("fart_create: FaceDetector instance is NULL after construction");
        delete handle;
        return nullptr;
    }

    g_debug("fart_create: created handle=%p instance=%p mode=%s anti_spoof=%s",
            handle,
            handle->instance,
            has_data_dir ? "file" : "json",
            anti_spoof_model_path.empty() ? "disabled" : "enabled");
    return handle;
}

void
fart_destroy(FaceAnalysisRecognition *handle)
{
    if (!handle) {
        g_debug("fart_destroy: handle is NULL");
        return;
    }

    if (!handle->instance) {
        g_debug("fart_destroy: handle->instance is NULL (handle=%p)", handle);
        delete handle;
        return;
    }

    delete handle->instance;
    handle->instance = nullptr;
    delete handle;

    g_debug("fart_destroy: destroyed handle");
}

static inline bool
validate_image_args(const unsigned char *image_data, int width, int height, int channels)
{
    if (!image_data) {
        g_debug("validate_image_args: image_data is NULL");
        return false;
    }

    if (width <= 0) {
        g_debug("validate_image_args: invalid width=%d", width);
        return false;
    }

    if (height <= 0) {
        g_debug("validate_image_args: invalid height=%d", height);
        return false;
    }

    if (!(channels == 1 || channels == 3 || channels == 4)) {
        g_debug("validate_image_args: invalid channels=%d (expected 1,3,4)", channels);
        return false;
    }

    return true;
}

Face *
fart_detect(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels)
{
    if (!handle) {
        g_debug("fart_detect: handle is NULL");
        return nullptr;
    }

    if (!handle->instance) {
        g_debug("fart_detect: handle->instance is NULL (handle=%p)", handle);
        return nullptr;
    }

    if (!validate_image_args(image_data, width, height, channels)) {
        g_debug("fart_detect: invalid image args (w=%d h=%d c=%d)", width, height, channels);
        return nullptr;
    }

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    if (image.empty()) {
        g_debug("fart_detect: constructed cv::Mat is empty (w=%d h=%d c=%d)", width, height, channels);
        return nullptr;
    }

    std::vector<FaceDetector::DetectedFace> detected = handle->instance->detect_faces(image);

    int count = (int)detected.size();
    if (count <= 0) {
        g_debug("fart_detect: no faces detected");
        return nullptr;
    }

    Face *faces_array = (Face *)std::malloc((size_t)count * sizeof(Face));
    if (!faces_array) {
        g_debug("fart_detect: malloc failed for %d faces", count);
        return nullptr;
    }

    for (int i = 0; i < count; ++i) {
        faces_array[i].x = detected[i].bbox.x;
        faces_array[i].y = detected[i].bbox.y;
        faces_array[i].width = detected[i].bbox.width;
        faces_array[i].height = detected[i].bbox.height;
        faces_array[i].confidence = detected[i].confidence;
        faces_array[i].face_count = count;
    }

    g_debug("fart_detect: returning %d faces", count);
    return faces_array;
}

void
fart_free_faces(Face *faces)
{
    if (!faces) {
        g_debug("fart_free_faces: faces is NULL");
        return;
    }

    std::free(faces);
    g_debug("fart_free_faces: freed faces array");
}

EnrollmentState
fart_enroll(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels,
            int *out_progress)
{
    if (out_progress)
        *out_progress = 0;

    if (!handle) {
        g_debug("fart_enroll: handle is NULL");
        return ENROLLMENT_FAIL;
    }

    if (!handle->instance) {
        g_debug("fart_enroll: handle->instance is NULL (handle=%p)", handle);
        return ENROLLMENT_FAIL;
    }

    if (!validate_image_args(image_data, width, height, channels)) {
        g_debug("fart_enroll: invalid image args (w=%d h=%d c=%d)", width, height, channels);
        return ENROLLMENT_FAIL;
    }

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    if (image.empty()) {
        g_debug("fart_enroll: constructed cv::Mat is empty (w=%d h=%d c=%d)", width, height, channels);
        return ENROLLMENT_FAIL;
    }

    EnrollmentState st = handle->instance->enroll_face(image, out_progress);

    if (out_progress)
        g_debug("fart_enroll: state=%d progress=%d", (int)st, *out_progress);
    else
        g_debug("fart_enroll: state=%d", (int)st);

    return st;
}

RecognitionState
fart_recognize(FaceAnalysisRecognition *handle,
               const unsigned char *image_data,
               int width, int height, int channels)
{
    if (!handle) {
        g_debug("fart_recognize: handle is NULL");
        return RECOGNITION_FAIL;
    }

    if (!handle->instance) {
        g_debug("fart_recognize: handle->instance is NULL (handle=%p)", handle);
        return RECOGNITION_FAIL;
    }

    if (!validate_image_args(image_data, width, height, channels)) {
        g_debug("fart_recognize: invalid image args (w=%d h=%d c=%d)", width, height, channels);
        return RECOGNITION_FAIL;
    }

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    if (image.empty()) {
        g_debug("fart_recognize: constructed cv::Mat is empty (w=%d h=%d c=%d)", width, height, channels);
        return RECOGNITION_FAIL;
    }

    RecognitionState st = handle->instance->recognize_face(image);
    g_debug("fart_recognize: %d", (int)st);
    return st;
}

int
fart_is_enrolled(FaceAnalysisRecognition *handle)
{
    if (!handle) {
        g_debug("fart_is_enrolled: handle is NULL");
        return 0;
    }

    if (!handle->instance) {
        g_debug("fart_is_enrolled: handle->instance is NULL (handle=%p)", handle);
        return 0;
    }

    int enrolled = handle->instance->is_enrolled() ? 1 : 0;
    g_debug("fart_is_enrolled: %d", enrolled);
    return enrolled;
}

char *
fart_export_enrollment_json(FaceAnalysisRecognition *handle)
{
    if (!handle) {
        g_debug("fart_export_enrollment_json: handle is NULL");
        return nullptr;
    }

    if (!handle->instance) {
        g_debug("fart_export_enrollment_json: handle->instance is NULL (handle=%p)", handle);
        return nullptr;
    }

    std::string enrollment_json = handle->instance->export_enrollment_json();
    char *out = (char *)std::malloc(enrollment_json.size() + 1);
    if (!out)
        return nullptr;

    std::memcpy(out, enrollment_json.c_str(), enrollment_json.size() + 1);
    return out;
}

int
fart_import_enrollment_json(FaceAnalysisRecognition *handle,
                            const char *enrollment_json)
{
    if (!handle) {
        g_debug("fart_import_enrollment_json: handle is NULL");
        return 0;
    }

    if (!handle->instance) {
        g_debug("fart_import_enrollment_json: handle->instance is NULL (handle=%p)", handle);
        return 0;
    }

    if (!enrollment_json) {
        g_debug("fart_import_enrollment_json: enrollment_json is NULL");
        return 0;
    }

    return handle->instance->import_enrollment_json(std::string(enrollment_json));
}

void
fart_free_string(char *str)
{
    if (!str) {
        g_debug("fart_free_string: str is NULL");
        return;
    }

    std::free(str);
}

} // extern "C"
