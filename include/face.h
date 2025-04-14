/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#ifndef FACE_H
#define FACE_H

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

/**
 * @brief Enrollment state.
 *
 * State values representing the current enrollment state.
 */
typedef enum {
    ENROLLMENT_FAIL = 0,          /**< Enrollment failed */
    ENROLLMENT_IN_PROGRESS,       /**< Enrollment in progress */
    ENROLLMENT_COMPLETE,          /**< Enrollment is complete */
    ENROLLMENT_SAVE_FAILED,       /**< Enrollment complete, but saving failed */
    ENROLLMENT_MULTIPLE_FACES,    /**< Multiple faces detected */
    ENROLLMENT_NO_FACE,           /**< No face detected */
    ENROLLMENT_BAD_LIGHTING       /**< Face detected but lighting conditions are poor */
} EnrollmentState;

/**
 * @brief Recognition state.
 *
 * state values representing the current recognition result.
 */
typedef enum {
    RECOGNITION_FAIL = 0,         /**< Recognition failed */
    RECOGNITION_NO_FACE,          /**< No face detected */
    RECOGNITION_MULTIPLE_FACES,   /**< Multiple faces detected */
    RECOGNITION_NOT_ENROLLED,     /**< No face has been enrolled */
    RECOGNITION_RECOGNIZED,       /**< The face is recognized */
    RECOGNITION_NOT_RECOGNIZED    /**< The face is not recognized */
} RecognitionState;

/*
 * Opaque handle for the internal FaceDetector object.
 */
typedef struct FacialUniversalRecognition FacialUniversalRecognition;

/*
 * Create and initialize a FaceDetector instance.
 * 'detection_model' and 'recognition_model' are paths to the TFLite model files.
 * Returns a valid handle on success or NULL on failure.
 */
FacialUniversalRecognition*
face_create(const char *detection_model, const char *recognition_model);

/*
 * Destroy the FaceDetector instance.
 */
void
face_destroy(FacialUniversalRecognition *handle);

/*
 * Detect faces in an image.
 * image_data should be in BGR format.
 * Returns an allocated array of Face structures (free with face_free_faces).
 */
Face*
face_detect(FacialUniversalRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels);

/*
 * Free the memory allocated by face_detect().
 */
void
face_free_faces(Face* faces);

/*
 * Enroll a face from an image.
 * If no face is enrolled, it performs enrollment.
 * Returns an EnrollmentState value.
 */
EnrollmentState
face_enroll(FacialUniversalRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels);

/*
 * Recognize a face from an image.
 * If no face is enrolled, returns RECOGNITION_NOT_ENROLLED.
 * Otherwise, returns a RecognitionState value.
 */
RecognitionState
face_recognize(FacialUniversalRecognition *handle,
               const unsigned char *image_data,
               int width, int height, int channels);

/*
 * Check if a face has been enrolled.
 * Returns 1 if enrolled; 0 otherwise.
 */
int
face_is_enrolled(FacialUniversalRecognition *handle);

#ifdef __cplusplus
}
#endif

#endif // FACE_H

