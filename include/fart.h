/*
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#ifndef FART_H
#define FART_H

#include "fart_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure representing a detected face.
 *
 * Contains location and confidence data for the detected face.
 */
typedef struct {
    int x;
    int y;
    int width;
    int height;
    float confidence;
    int face_count;
} Face;

typedef struct FaceAnalysisRecognition FaceAnalysisRecognition;

/*
 * Create and initialize a FaceDetector instance.
 * 'detection_model' and 'recognition_model' are paths to the TFLite model files.
 * Returns a valid handle on success or NULL on failure.
 */
FaceAnalysisRecognition *
fart_create(const char *detection_model, const char *recognition_model);

/*
 * Destroy the FaceDetector instance.
 */
void
fart_destroy(FaceAnalysisRecognition *handle);

/*
 * Detect faces in an image.
 * image_data should be in BGR format.
 * Returns an allocated array of Face structures (free with fart_free_faces).
 */
Face *
fart_detect(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels);

/*
 * Free the memory allocated by fart_detect().
 */
void
fart_free_faces(Face *faces);

/*
 * Enroll a face from an image.
 * If no face is enrolled, it performs enrollment.
 * Returns an EnrollmentState value.
 */
EnrollmentState
fart_enroll(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels);

/*
 * Recognize a face from an image.
 * If no face is enrolled, returns RECOGNITION_NOT_ENROLLED.
 * Otherwise, returns a RecognitionState value.
 */
RecognitionState
fart_recognize(FaceAnalysisRecognition *handle,
               const unsigned char *image_data,
               int width, int height, int channels);

/*
 * Check if a face has been enrolled.
 * Returns 1 if enrolled; 0 otherwise.
 */
int
fart_is_enrolled(FaceAnalysisRecognition *handle);

#ifdef __cplusplus
}
#endif

#endif // FART_H
