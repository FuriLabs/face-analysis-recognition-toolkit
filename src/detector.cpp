/*
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#include "detector.h"
#include "gbinder_helper.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <glib.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate_c_api.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

FaceDetector::FaceDetector(const std::string& detection_model_path,
                           const std::string& recognition_model_path,
                           const char *data_dir,
                           const char *enrollment_json,
                           float min_confidence,
                           float max_distance)
    : min_confidence_(min_confidence),
      max_distance_(max_distance)
{
    gboolean has_data_dir = data_dir && data_dir[0] != '\0';
    gboolean has_json = enrollment_json != nullptr;

    if (has_data_dir == has_json)
        throw std::runtime_error("Exactly one of data_dir or enrollment_json must be provided");

    file_storage_enabled_ = has_data_dir ? true : false;

    if (file_storage_enabled_) {
        g_debug("Initializing FaceDetector in file storage mode...");

        data_dir_ = fs::path(data_dir);
        if (data_dir_.empty())
            throw std::runtime_error("data_dir is empty");
    } else {
        g_debug("Initializing FaceDetector in JSON storage mode...");
        data_dir_.clear();
    }

    init_common(detection_model_path, recognition_model_path);

    if (file_storage_enabled_) {
        int load_status = load_enrolled_face();
        fs::path enrollment_file = get_data_dir() / "enrolled_face.json";
        if (fs::exists(enrollment_file) && load_status == 0)
            throw std::runtime_error("Failed to load enrolled face data");
    } else if (enrollment_json && enrollment_json[0] != '\0') {
        if (!import_enrollment_json(std::string(enrollment_json)))
            throw std::runtime_error("Failed to import enrollment JSON");
    }
}

void
FaceDetector::init_common(const std::string& detection_model_path,
                          const std::string& recognition_model_path)
{
    if (!find_hal("android.hardware.neuralnetworks"))
        throw std::runtime_error("Neural Networks HAL not found in hwservicemanager");

    detection_model_ = tflite::FlatBufferModel::BuildFromFile(detection_model_path.c_str());
    if (!detection_model_)
        throw std::runtime_error("Failed to load detection model");

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*detection_model_, resolver)(&detector_);
    if (!detector_)
        throw std::runtime_error("Failed to build detection interpreter");

    TfLiteNnapiDelegateOptions nnapi_options = TfLiteNnapiDelegateOptionsDefault();
    nnapi_options.allow_fp16 = true;
    nnapi_options.disallow_nnapi_cpu = false;

    nnapi_detection_delegate_ = TfLiteNnapiDelegateCreate(&nnapi_options);
    if (!nnapi_detection_delegate_) {
        g_debug("Failed to create NNAPI delegate for detection model, falling back to CPU");
    } else {
        g_debug("Using NNAPI delegate for detection model");
        if (detector_->ModifyGraphWithDelegate(nnapi_detection_delegate_) != kTfLiteOk) {
            g_debug("Failed to apply NNAPI delegate to detection model, falling back to CPU");
            TfLiteNnapiDelegateDelete(nnapi_detection_delegate_);
            nnapi_detection_delegate_ = nullptr;
        }
    }

    if (detector_->AllocateTensors() != kTfLiteOk)
        throw std::runtime_error("Failed to allocate tensors for detection model");

    recognition_model_ = tflite::FlatBufferModel::BuildFromFile(recognition_model_path.c_str());
    if (!recognition_model_)
        throw std::runtime_error("Failed to load recognition model");

    tflite::InterpreterBuilder(*recognition_model_, resolver)(&recognizer_);
    if (!recognizer_)
        throw std::runtime_error("Failed to build recognition interpreter");

    nnapi_recognition_delegate_ = TfLiteNnapiDelegateCreate(&nnapi_options);
    if (!nnapi_recognition_delegate_) {
        g_debug("Failed to create NNAPI delegate for recognition model, falling back to CPU");
    } else {
        g_debug("Using NNAPI delegate for recognition model");
        if (recognizer_->ModifyGraphWithDelegate(nnapi_recognition_delegate_) != kTfLiteOk) {
            g_debug("Failed to apply NNAPI delegate to recognition model, falling back to CPU");
            TfLiteNnapiDelegateDelete(nnapi_recognition_delegate_);
            nnapi_recognition_delegate_ = nullptr;
        }
    }

    if (recognizer_->AllocateTensors() != kTfLiteOk)
        throw std::runtime_error("Failed to allocate tensors for recognition model");

    g_debug("Detector inputs: %d", (int)detector_->inputs().size());
    g_debug("Detector outputs: %d", (int)detector_->outputs().size());

    detect_input_index_ = detector_->inputs()[0];
    detect_output_boxes_ = detector_->outputs()[0];
    detect_output_classes_ = detector_->outputs()[1];
    detect_output_scores_ = detector_->outputs()[2];

    g_debug("Recognizer inputs: %d", (int)recognizer_->inputs().size());
    g_debug("Recognizer outputs: %d", (int)recognizer_->outputs().size());

    recog_input_index_ = recognizer_->inputs()[0];
    recog_output_index_ = recognizer_->outputs()[0];
}

FaceDetector::~FaceDetector()
{
    if (nnapi_detection_delegate_) {
        TfLiteNnapiDelegateDelete(nnapi_detection_delegate_);
        nnapi_detection_delegate_ = nullptr;
    }
    if (nnapi_recognition_delegate_) {
        TfLiteNnapiDelegateDelete(nnapi_recognition_delegate_);
        nnapi_recognition_delegate_ = nullptr;
    }
}

std::vector<FaceDetector::DetectedFace>
FaceDetector::detect_faces(const cv::Mat& image)
{
    if (image.empty())
        return {};

    int height = image.rows;
    int width = image.cols;

    std::lock_guard<std::mutex> lock(detector_mutex_);

    try {
        cv::Mat resized_image;
        cv::resize(image, resized_image, cv::Size(FD_INPUT_SIZE, FD_INPUT_SIZE));

        if (resized_image.channels() == 1)
            cv::cvtColor(resized_image, resized_image, cv::COLOR_GRAY2RGB);
        else if (resized_image.channels() == 4)
            cv::cvtColor(resized_image, resized_image, cv::COLOR_BGRA2RGB);
        else if (resized_image.channels() == 3)
            cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

        g_debug("Running detection inference...");
        int input_index = detector_->inputs()[0];
        TfLiteTensor *input_tensor = detector_->tensor(input_index);
        if (!input_tensor) {
            g_debug("Failed to get input tensor");
            return {};
        }

        uint8_t *input_data = input_tensor->data.uint8;
        if (!input_data) {
            g_debug("Input tensor data is null");
            return {};
        }

        std::memcpy(input_data,
                    resized_image.data,
                    resized_image.total() * resized_image.elemSize());

        g_debug("Running inference with %s...",
                nnapi_detection_delegate_ ? "NNAPI delegate" : "CPU");
        if (detector_->Invoke() != kTfLiteOk) {
            g_debug("Failed to invoke detection model");
            return {};
        }

        int boxes_index  = detector_->outputs()[0];
        int scores_index = detector_->outputs()[2];

        TfLiteTensor *boxes_tensor  = detector_->tensor(boxes_index);
        TfLiteTensor *scores_tensor = detector_->tensor(scores_index);

        if (!boxes_tensor || !scores_tensor) {
            g_debug("Failed to get output tensors");
            return {};
        }

        const float *boxes_data  = reinterpret_cast<const float *>(boxes_tensor->data.data);
        const float *scores_data = reinterpret_cast<const float *>(scores_tensor->data.data);

        if (!boxes_data || !scores_data) {
            g_debug("Output tensor data is null");
            return {};
        }

        int detection_count = 10;
        if (scores_tensor->dims && scores_tensor->dims->size > 0) {
            int last_dim = scores_tensor->dims->data[scores_tensor->dims->size - 1];
            if (last_dim > 0)
                detection_count = last_dim;
        }

        g_debug("Detection count from tensor: %d", detection_count);

        std::vector<DetectedFace> detected_faces;
        for (int i = 0; i < detection_count; i++) {
            if (scores_data[i] >= min_confidence_) {
                /* i don't even know at this point */
                float ymin = boxes_data[i * 4];
                float xmin = boxes_data[i * 4 + 1];
                float ymax = boxes_data[i * 4 + 2];
                float xmax = boxes_data[i * 4 + 3];

                int xmin_px = (int)(xmin * width);
                int xmax_px = (int)(xmax * width);
                int ymin_px = (int)(ymin * height);
                int ymax_px = (int)(ymax * height);

                cv::Rect rect(xmin_px, ymin_px,
                              xmax_px - xmin_px,
                              ymax_px - ymin_px);
                rect &= cv::Rect(0, 0, width, height);

                if (rect.width <= 0 || rect.height <= 0) {
                    g_debug("Ignoring invalid face rect: [%d, %d, %d, %d] score=%f",
                            xmin_px, ymin_px, xmax_px, ymax_px, scores_data[i]);
                    continue;
                }

                g_debug("Face detected: [%d, %d, %d, %d] score=%f",
                        rect.x, rect.y, rect.x + rect.width, rect.y + rect.height, scores_data[i]);

                DetectedFace face;
                face.bbox = rect;
                face.confidence = scores_data[i];
                detected_faces.push_back(face);
            }
        }

        g_debug("Detected faces count: %zu", detected_faces.size());
        return detected_faces;
    } catch (const std::exception& e) {
        g_debug("Exception in detect_faces: %s", e.what());
        return {};
    }
}

std::vector<float>
FaceDetector::get_face_embedding(const cv::Mat& face_image)
{
    if (face_image.empty())
        return {};

    std::lock_guard<std::mutex> lock(recognizer_mutex_);

    try {
        cv::Mat processed_image;
        cv::resize(face_image, processed_image, cv::Size(FR_INPUT_SIZE, FR_INPUT_SIZE));

        if (processed_image.channels() == 1)
            cv::cvtColor(processed_image, processed_image, cv::COLOR_GRAY2RGB);
        else if (processed_image.channels() == 4)
            cv::cvtColor(processed_image, processed_image, cv::COLOR_BGRA2RGB);
        else if (processed_image.channels() == 3)
            cv::cvtColor(processed_image, processed_image, cv::COLOR_BGR2RGB);

        if (FR_IS_QUANTIZED) {
            uint8_t *input = recognizer_->typed_input_tensor<uint8_t>(recog_input_index_);
            if (!input)
                return {};

            std::memcpy(input,
                        processed_image.data,
                        processed_image.total() * processed_image.elemSize());
        } else {
            float *input = recognizer_->typed_input_tensor<float>(recog_input_index_);
            if (!input)
                return {};

            cv::Mat float_image;
            processed_image.convertTo(float_image, CV_32F);
            float_image = (float_image - 127.5f) / 127.5f;

            std::memcpy(input,
                        float_image.data,
                        float_image.total() * float_image.elemSize());
        }

        g_debug("Running recognition with %s...",
                nnapi_recognition_delegate_ ? "NNAPI delegate" : "CPU");
        if (recognizer_->Invoke() != kTfLiteOk)
            return {};

        TfLiteTensor *output_tensor = recognizer_->tensor(recog_output_index_);
        if (!output_tensor)
            return {};

        const float *output_data = reinterpret_cast<const float *>(output_tensor->data.data);
        if (!output_data)
            return {};

        int embedding_size = output_tensor->dims->data[1];
        return std::vector<float>(output_data, output_data + embedding_size);
    } catch (const std::exception& e) {
        g_debug("Exception in get_face_embedding: %s", e.what());
        return {};
    }
}

cv::Mat
FaceDetector::extract_face(const cv::Mat& image, const cv::Rect& bbox)
{
    cv::Rect safe_rect = bbox & cv::Rect(0, 0, image.cols, image.rows);
    if (safe_rect.width <= 0 || safe_rect.height <= 0)
        return cv::Mat();

    return image(safe_rect);
}

int
FaceDetector::check_brightness(const cv::Mat& face_image)
{
    if (face_image.empty())
        return 0;

    cv::Mat gray;
    if (face_image.channels() == 1)
        gray = face_image;
    else if (face_image.channels() == 4)
        cv::cvtColor(face_image, gray, cv::COLOR_BGRA2GRAY);
    else
        cv::cvtColor(face_image, gray, cv::COLOR_BGR2GRAY);

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray, mean, stddev);

    double brightness = mean[0];
    double contrast = stddev[0];

    int total_pixels = gray.rows * gray.cols;
    if (total_pixels <= 0)
        return 0;

    cv::Mat dark_mask;
    cv::Mat bright_mask;
    cv::compare(gray, 35, dark_mask, cv::CMP_LT);
    cv::compare(gray, 240, bright_mask, cv::CMP_GT);

    int dark_pixels = cv::countNonZero(dark_mask);
    int bright_pixels = cv::countNonZero(bright_mask);

    double dark_ratio = (double)dark_pixels / (double)total_pixels;
    double bright_ratio = (double)bright_pixels / (double)total_pixels;

    g_debug("Lighting check: brightness=%f contrast=%f dark_ratio=%f bright_ratio=%f",
            brightness, contrast, dark_ratio, bright_ratio);

    if (brightness < 45.0 || dark_ratio > 0.45)
        return -1; // Bad lighting

    if (brightness > 210.0 || bright_ratio > 0.35)
        return -1; // Bad lighting

    if (contrast < 25.0)
        return 0; // Suboptimal

    return 1; // Optimal
}

float
FaceDetector::compare_embeddings(const std::vector<float>& embedding1,
                                 const std::vector<float>& embedding2)
{
    if (embedding1.empty() || embedding2.empty())
        return std::numeric_limits<float>::max();

    if (embedding1.size() != embedding2.size())
        return std::numeric_limits<float>::max();

    std::vector<float> norm1 = normalize_vector(embedding1);
    std::vector<float> norm2 = normalize_vector(embedding2);

    float sum_sq = 0.0f;
    for (size_t i = 0; i < norm1.size(); ++i) {
        float diff = norm1[i] - norm2[i];
        sum_sq += diff * diff;
    }

    return std::sqrt(sum_sq);
}

float
FaceDetector::get_max_distance() const
{
    return max_distance_;
}

EnrollmentState
FaceDetector::enroll_face(const cv::Mat& frame, int *out_progress)
{
    if (out_progress)
        *out_progress = 0;

    std::vector<DetectedFace> faces = detect_faces(frame);
    if (faces.size() != 1) {
        if (faces.size() > 1) {
            g_debug("Multiple faces detected");
            return ENROLLMENT_MULTIPLE_FACES;
        } else {
            g_debug("No face detected");
            return ENROLLMENT_NO_FACE;
        }
    }

    cv::Mat face_img = extract_face(frame, faces[0].bbox);
    if (face_img.empty()) {
        g_debug("Invalid face image extracted");
        return ENROLLMENT_FAIL;
    }

    int brightness = check_brightness(face_img);
    if (brightness < 1) {
        g_debug((brightness == -1) ? "Poor lighting conditions" : "Suboptimal lighting");
        return ENROLLMENT_BAD_LIGHTING;
    }

    std::vector<float> embedding = get_face_embedding(face_img);
    if (embedding.empty()) {
        g_debug("Failed to generate face embedding");
        return ENROLLMENT_FAIL;
    }

    pending_enrollments_.push_back(embedding);
    int count = (int)pending_enrollments_.size();

    if (count >= 10) {
        std::vector<float> avg(embedding.size(), 0.0f);
        for (auto& emb : pending_enrollments_) {
            for (size_t i = 0; i < emb.size(); i++)
                avg[i] += emb[i];
        }

        for (size_t i = 0; i < avg.size(); i++)
            avg[i] /= (float)count;

        enrolled_embedding_ = normalize_vector(avg);
        pending_enrollments_.clear();

        int save_status = save_enrolled_face();
        if (!save_status) {
            g_debug("Enrollment complete, but failed to save enrolled face");
            if (out_progress)
                *out_progress = 0;
            return ENROLLMENT_SAVE_FAILED;
        }

        if (out_progress)
            *out_progress = 100;

        g_debug("Enrollment complete");
        return ENROLLMENT_COMPLETE;
    } else {
        int progress = (count * 100) / 10;
        if (progress < 0)
            progress = 0;
        if (progress > 99)
            progress = 99;

        if (out_progress)
            *out_progress = progress;

        g_debug("Enrollment progress: %d%%", progress);
        return ENROLLMENT_IN_PROGRESS;
    }
}

RecognitionState
FaceDetector::recognize_face(const cv::Mat& frame)
{
    std::vector<DetectedFace> faces = detect_faces(frame);
    if (faces.size() != 1) {
        if (faces.size() > 1) {
            g_debug("Multiple faces detected");
            return RECOGNITION_MULTIPLE_FACES;
        } else {
            g_debug("No face detected");
            return RECOGNITION_NO_FACE;
        }
    }

    cv::Mat face_img = extract_face(frame, faces[0].bbox);
    if (face_img.empty()) {
        g_debug("Invalid face image extracted");
        return RECOGNITION_FAIL;
    }

    std::vector<float> embedding = get_face_embedding(face_img);
    if (embedding.empty()) {
        g_debug("Failed to generate face embedding");
        return RECOGNITION_FAIL;
    }

    if (enrolled_embedding_.empty()) {
        g_debug("No face has been enrolled");
        return RECOGNITION_NOT_ENROLLED;
    }

    float distance = compare_embeddings(embedding, enrolled_embedding_);
    bool recognized = distance < get_max_distance();
    g_debug("Distance: %.3f, %s", distance, recognized ? "Recognized" : "Not recognized");
    return recognized ? RECOGNITION_RECOGNIZED : RECOGNITION_NOT_RECOGNIZED;
}

bool
FaceDetector::is_enrolled() const
{
    return !enrolled_embedding_.empty();
}

std::vector<float>
FaceDetector::normalize_vector(const std::vector<float>& vec)
{
    float norm = 0.0f;
    for (float val : vec)
        norm += val * val;

    norm = std::sqrt(norm);
    if (norm <= 0.0f)
        return vec;

    std::vector<float> normalized(vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        normalized[i] = vec[i] / norm;

    return normalized;
}

std::string
FaceDetector::export_enrollment_json() const
{
    try {
        if (enrolled_embedding_.empty())
            return std::string();

        json j = enrolled_embedding_;
        return j.dump();
    } catch (const std::exception& e) {
        g_debug("Error exporting enrollment JSON: %s", e.what());
        return std::string();
    }
}

int
FaceDetector::import_enrollment_json(const std::string& enrollment_json)
{
    try {
        if (enrollment_json.empty()) {
            enrolled_embedding_.clear();
            return 1;
        }

        json j = json::parse(enrollment_json);
        enrolled_embedding_ = normalize_vector(j.get<std::vector<float>>());
        g_debug("Imported enrolled face from JSON string");
        return 1;
    } catch (const std::exception& e) {
        g_debug("Error importing enrollment JSON: %s", e.what());
        enrolled_embedding_.clear();
        return 0;
    }
}

int
FaceDetector::save_enrolled_face()
{
    if (!file_storage_enabled_) {
        g_debug("Enrollment stored in memory only; caller must export JSON");
        return 1;
    }

    try {
        json j = enrolled_embedding_;
        fs::path data_dir = get_data_dir();
        fs::create_directories(data_dir);
        fs::path enrollment_file = data_dir / "enrolled_face.json";

        std::ofstream file(enrollment_file);
        if (!file.is_open())
            return 0;

        file << j;
        file.close();
        g_debug("Enrollment saved to %s", enrollment_file.string().c_str());
        return 1;
    } catch (const std::exception& e) {
        g_debug("Error saving enrollment: %s", e.what());
        return 0;
    }
}

int
FaceDetector::load_enrolled_face()
{
    if (!file_storage_enabled_)
        return 1;

    try {
        fs::path data_dir = get_data_dir();
        fs::path enrollment_file = data_dir / "enrolled_face.json";

        if (!fs::exists(enrollment_file))
            return 1;

        std::ifstream file(enrollment_file);
        if (!file.is_open())
            return 0;

        json j;
        file >> j;
        enrolled_embedding_ = normalize_vector(j.get<std::vector<float>>());
        file.close();
        g_debug("Loaded enrolled face from %s", enrollment_file.string().c_str());
        return 1;
    } catch (const std::exception& e) {
        g_debug("Error loading enrolled face: %s", e.what());
        enrolled_embedding_.clear();
        return 0;
    }
}

fs::path
FaceDetector::get_data_dir() const
{
    return data_dir_;
}
