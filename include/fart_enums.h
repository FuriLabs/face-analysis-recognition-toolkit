/*
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#ifndef FART_ENUMS_H
#define FART_ENUMS_H

#ifdef __cplusplus
extern "C" {
#endif

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
 * State values representing the current recognition result.
 */
typedef enum {
    RECOGNITION_FAIL = 0,         /**< Recognition failed */
    RECOGNITION_NO_FACE,          /**< No face detected */
    RECOGNITION_MULTIPLE_FACES,   /**< Multiple faces detected */
    RECOGNITION_NOT_ENROLLED,     /**< No face has been enrolled */
    RECOGNITION_RECOGNIZED,       /**< The face is recognized */
    RECOGNITION_NOT_RECOGNIZED    /**< The face is not recognized */
} RecognitionState;

#ifdef __cplusplus
}
#endif

#endif // FART_ENUMS_H
