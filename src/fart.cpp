/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "fart.h"
#include "detector.h"

struct FaceAnalysisRecognition {
    FaceDetector *instance = nullptr;
};

extern "C" {

FaceAnalysisRecognition *
fart_create(const char *detection_model, const char *recognition_model)
{
    if (!detection_model || !recognition_model)
        return nullptr;

    FaceAnalysisRecognition *handle = new FaceAnalysisRecognition();

    try {
        handle->instance = new FaceDetector(std::string(detection_model),
                                            std::string(recognition_model));
    } catch (const std::exception &e) {
        std::cerr << "Error in fart_create: " << e.what() << std::endl;
        delete handle;
        return nullptr;
    }

    return handle;
}

void
fart_destroy(FaceAnalysisRecognition *handle)
{
    if (!handle)
        return;

    delete handle->instance;
    handle->instance = nullptr;
    delete handle;
}

static inline bool
validate_image_args(const unsigned char *image_data, int width, int height, int channels)
{
    return image_data && width > 0 && height > 0 && (channels == 1 || channels == 3 || channels == 4);
}

Face *
fart_detect(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels)
{
    if (!handle || !handle->instance || !validate_image_args(image_data, width, height, channels))
        return nullptr;

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    std::vector<FaceDetector::DetectedFace> detected = handle->instance->detect_faces(image);

    int count = (int)detected.size();
    if (count <= 0)
        return nullptr;

    Face *faces_array = (Face *)std::malloc((size_t)count * sizeof(Face));
    if (!faces_array)
        return nullptr;

    for (int i = 0; i < count; ++i) {
        faces_array[i].x = detected[i].bbox.x;
        faces_array[i].y = detected[i].bbox.y;
        faces_array[i].width = detected[i].bbox.width;
        faces_array[i].height = detected[i].bbox.height;
        faces_array[i].confidence = detected[i].confidence;
        faces_array[i].face_count = count;
    }

    return faces_array;
}

void
fart_free_faces(Face *faces)
{
    if (faces)
        std::free(faces);
}

EnrollmentState
fart_enroll(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels)
{
    if (!handle || !handle->instance || !validate_image_args(image_data, width, height, channels))
        return ENROLLMENT_FAIL;

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    return handle->instance->enroll_face(image);
}

RecognitionState
fart_recognize(FaceAnalysisRecognition *handle,
               const unsigned char *image_data,
               int width, int height, int channels)
{
    if (!handle || !handle->instance || !validate_image_args(image_data, width, height, channels))
        return RECOGNITION_FAIL;

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void *)image_data);

    return handle->instance->recognize_face(image);
}

int
fart_is_enrolled(FaceAnalysisRecognition *handle)
{
    if (!handle || !handle->instance)
        return 0;

    return handle->instance->is_enrolled() ? 1 : 0;
}

} // extern "C"
