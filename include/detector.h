/*
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#ifndef DETECTOR_H
#define DETECTOR_H

#include <memory>
#include <mutex>
#include <vector>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

#include "fart.h"

class FaceDetector {
public:
    /* should be fine tuned */
    static constexpr int  FD_INPUT_SIZE = 300;
    static constexpr bool FD_IS_QUANTIZED = true;

    static constexpr int  FR_INPUT_SIZE = 112;
    static constexpr bool FR_IS_QUANTIZED = false;

    struct DetectedFace {
        cv::Rect bbox;
        float confidence;
    };

    FaceDetector(const std::string& detection_model_path,
                 const std::string& recognition_model_path,
                 const char *data_dir,
                 const char *enrollment_json,
                 float min_confidence = 0.5f,
                 float max_distance = 0.7f);

    ~FaceDetector();

    std::vector<DetectedFace>
    detect_faces(const cv::Mat& image);

    std::vector<float>
    get_face_embedding(const cv::Mat& face_image);

    cv::Mat
    extract_face(const cv::Mat& image, const cv::Rect& bbox);

    int
    check_brightness(const cv::Mat& face_image);

    float
    compare_embeddings(const std::vector<float>& embedding1,
                       const std::vector<float>& embedding2);

    float
    get_max_distance() const;

    EnrollmentState
    enroll_face(const cv::Mat& frame, int *out_progress);

    RecognitionState
    recognize_face(const cv::Mat& frame);

    bool
    is_enrolled() const;

    std::string
    export_enrollment_json() const;

    int
    import_enrollment_json(const std::string& enrollment_json);

private:
    std::unique_ptr<tflite::FlatBufferModel> detection_model_;
    std::unique_ptr<tflite::Interpreter> detector_;
    int detect_input_index_ = -1;
    int detect_output_boxes_ = -1;
    int detect_output_classes_ = -1;
    int detect_output_scores_ = -1;

    std::unique_ptr<tflite::FlatBufferModel> recognition_model_;
    std::unique_ptr<tflite::Interpreter> recognizer_;
    int recog_input_index_ = -1;
    int recog_output_index_ = -1;

    TfLiteDelegate* nnapi_detection_delegate_ = nullptr;
    TfLiteDelegate* nnapi_recognition_delegate_ = nullptr;

    float min_confidence_ = 0.5f;
    float max_distance_ = 0.7f;

    std::mutex detector_mutex_;
    std::mutex recognizer_mutex_;

    std::vector<std::vector<float>> pending_enrollments_;
    std::vector<float> enrolled_embedding_;

    std::filesystem::path data_dir_;
    bool file_storage_enabled_ = false;

    void
    init_common(const std::string& detection_model_path,
                const std::string& recognition_model_path);

    std::vector<float>
    normalize_vector(const std::vector<float>& vec);

    int
    save_enrolled_face();

    int
    load_enrolled_face();

    std::filesystem::path
    get_data_dir() const;
};

#endif // DETECTOR_H
