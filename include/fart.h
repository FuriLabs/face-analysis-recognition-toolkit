/*
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2026 Bardia Moshiri <bardia@furilabs.com>
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
 * 'anti_spoof_model' is an optional path to the anti-spoof TFLite model file.
 * Pass NULL or an empty string to disable anti-spoof checking.
 * 'data_dir' is the directory used for persistent face data (enrolled_face.json).
 * 'enrollment_json' is optional in-memory enrollment data.
 *
 * Exactly one mode must be selected:
 *   - data_dir != NULL and enrollment_json == NULL: file-based mode
 *   - data_dir == NULL and enrollment_json != NULL: JSON/in-memory mode
 *
 * Returns a valid handle on success or NULL on failure.
 */
FaceAnalysisRecognition *
fart_create(const char *detection_model,
            const char *recognition_model,
            const char *anti_spoof_model,
            const char *data_dir,
            const char *enrollment_json);

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
 * out_progress (optional) receives [0..100] enrollment percentage.
 * Returns an EnrollmentState value.
 */
EnrollmentState
fart_enroll(FaceAnalysisRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels,
            int *out_progress);

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

/*
 * Export current enrolled face data as a JSON string.
 * Caller must free the returned string with fart_free_string().
 * Returns NULL on failure or if handle is invalid.
 */
char *
fart_export_enrollment_json(FaceAnalysisRecognition *handle);

/*
 * Import enrolled face data from a JSON string.
 * Returns 1 on success; 0 otherwise.
 */
int
fart_import_enrollment_json(FaceAnalysisRecognition *handle,
                            const char *enrollment_json);

/*
 * Free a string returned by libfart.
 */
void
fart_free_string(char *str);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FART_H
