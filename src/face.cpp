/**
 * SPDX-License-Identifier: GPL-2.0-only
 * Copyright (C) 2025 Bardia Moshiri <bardia@furilabs.com>
 */

#include <fstream>

#include <glib.h>

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/kernels/register.h"

#include "face.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

class FaceDetector {
public:
    /* should be fine tuned */
    static constexpr int FD_INPUT_SIZE = 300;
    static constexpr bool FD_IS_QUANTIZED = true;

    static constexpr int FR_INPUT_SIZE = 112;
    static constexpr bool FR_IS_QUANTIZED = false;

    struct Face {
        cv::Rect bbox;
        float confidence;
    };

    FaceDetector(const std::string &detection_model_path,
                 const std::string &recognition_model_path,
                 float min_confidence = 0.5f,
                 float max_distance = 0.7f)
        : min_confidence_(min_confidence), max_distance_(max_distance)
    {
        g_debug("Initializing FaceDetector...");

        detection_model_ = tflite::FlatBufferModel::BuildFromFile(detection_model_path.c_str());
        if (!detection_model_)
            throw std::runtime_error("Failed to load detection model");

        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder(*detection_model_, resolver)(&detector_);
        if (!detector_)
            throw std::runtime_error("Failed to build detection interpreter");

        if (detector_->AllocateTensors() != kTfLiteOk)
            throw std::runtime_error("Failed to allocate tensors for detection model");

        recognition_model_ = tflite::FlatBufferModel::BuildFromFile(recognition_model_path.c_str());
        if (!recognition_model_)
            throw std::runtime_error("Failed to load recognition model");

        tflite::InterpreterBuilder(*recognition_model_, resolver)(&recognizer_);
        if (!recognizer_)
            throw std::runtime_error("Failed to build recognition interpreter");

        if (recognizer_->AllocateTensors() != kTfLiteOk)
            throw std::runtime_error("Failed to allocate tensors for recognition model");

        g_debug("Detector inputs: %d", detector_->inputs().size());
        g_debug("Detector outputs: %d", detector_->outputs().size());

        detect_input_index_ = detector_->inputs()[0];
        detect_output_boxes_ = detector_->outputs()[0];
        detect_output_classes_ = detector_->outputs()[1];
        detect_output_scores_ = detector_->outputs()[2];

        g_debug("Recognizer inputs: %d", recognizer_->inputs().size());
        g_debug("Recognizer outputs: %d", recognizer_->outputs().size());

        recog_input_index_ = recognizer_->inputs()[0];
        recog_output_index_ = recognizer_->outputs()[0];

        create_brightness_test();

        int load_status = load_enrolled_face();
        fs::path enrollment_file = get_data_dir() / "enrolled_face.json";
        if (fs::exists(enrollment_file) && load_status == 0)
            throw std::runtime_error("Failed to load enrolled face data");
    }

    std::vector<Face>
    detect_faces(const cv::Mat &image)
    {
        if (image.empty())
            return {};

        int height = image.rows, width = image.cols;

        std::lock_guard<std::mutex> lock(detector_mutex_);

        try {
            /* incorrect to hell and back. but it works! */
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

            g_debug("Input tensor found");
            uint8_t *input_data = input_tensor->data.uint8;
            if (!input_data) {
                g_debug("Input tensor data is null");
                return {};
            }

            std::memcpy(input_data, resized_image.data, resized_image.total() * resized_image.elemSize());

            g_debug("Running inference...");
            if (detector_->Invoke() != kTfLiteOk) {
                g_debug("Failed to invoke detection model");
                return {};
            }

            int boxes_index = detector_->outputs()[0];
            int scores_index = detector_->outputs()[2];

            TfLiteTensor *boxes_tensor = detector_->tensor(boxes_index);
            TfLiteTensor *scores_tensor = detector_->tensor(scores_index);

            if (!boxes_tensor || !scores_tensor) {
                g_debug("Failed to get output tensors");
                return {};
            }

            const float *boxes_data = reinterpret_cast<const float*>(boxes_tensor->data.data);
            const float *scores_data = reinterpret_cast<const float*>(scores_tensor->data.data);
            if (!boxes_data || !scores_data) {
                g_debug("Output tensor data is null");
                return {};
            }

            std::vector<Face> detected_faces;
            for (int i = 0; i < 10; i++) {
                if (scores_data[i] >= min_confidence_) {
                    /* i don't even know at this point */
                    float ymin = boxes_data[i * 4];
                    float xmin = boxes_data[i * 4 + 1];
                    float ymax = boxes_data[i * 4 + 2];
                    float xmax = boxes_data[i * 4 + 3];

                    int xmin_px = static_cast<int>(xmin * width);
                    int xmax_px = static_cast<int>(xmax * width);
                    int ymin_px = static_cast<int>(ymin * height);
                    int ymax_px = static_cast<int>(ymax * height);

                    g_debug("Face detected: [%d, %d, %d, %d] score=%f",
                            xmin_px, ymin_px, xmax_px, ymax_px, scores_data[i]);

                    Face face;
                    face.bbox = cv::Rect(xmin_px, ymin_px, xmax_px - xmin_px, ymax_px - ymin_px);
                    face.confidence = scores_data[i];
                    detected_faces.push_back(face);
                }
            }

            g_debug("Detected faces count: %zu", detected_faces.size());
            return detected_faces;
        } catch (const std::exception &e) {
            g_debug("Exception in detect_faces:", e.what());
            return {};
        }
    }

    std::vector<float>
    get_face_embedding(const cv::Mat &face_image)
    {
        if (face_image.empty())
            return std::vector<float>();

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
                    return std::vector<float>();

                std::memcpy(input, processed_image.data,
                            processed_image.total() * processed_image.elemSize());
            } else {
                float *input = recognizer_->typed_input_tensor<float>(recog_input_index_);
                if (!input)
                    return std::vector<float>();

                cv::Mat float_image;
                processed_image.convertTo(float_image, CV_32F);
                float_image = (float_image - 127.5f) / 127.5f;
                std::memcpy(input, float_image.data,
                            float_image.total() * float_image.elemSize());
            }

            if (recognizer_->Invoke() != kTfLiteOk)
                return std::vector<float>();

            TfLiteTensor *output_tensor = recognizer_->tensor(recog_output_index_);
            if (!output_tensor)
                return std::vector<float>();

            const float *output_data = reinterpret_cast<const float*>(output_tensor->data.data);
            if (!output_data)
                return std::vector<float>();

            int embedding_size = output_tensor->dims->data[1];

            return std::vector<float>(output_data, output_data + embedding_size);
        } catch (const std::exception &) {
            return std::vector<float>();
        }
    }

    cv::Mat
    extract_face(const cv::Mat &image, const cv::Rect &bbox)
    {
        cv::Rect safe_rect = bbox & cv::Rect(0, 0, image.cols, image.rows);
        if (safe_rect.width <= 0 || safe_rect.height <= 0)
            return cv::Mat();

        return image(safe_rect);
    }

    /* likely very broken */
    int
    check_brightness(const std::vector<float>& embedding)
    {
        if (embedding.empty())
            return 0;

        float white_distance = compare_embeddings(embedding, brightness_test_white_);
        float black_distance = compare_embeddings(embedding, brightness_test_black_);
        g_debug("Brightness test - White: %f, Black: %f", white_distance, black_distance);
        if (white_distance < 0.5 || black_distance < 0.4)
            return -1;  // Bad lighting
        else if (white_distance + black_distance < 2.2)
            return 0;   // Suboptimal
        else
            return 1;   // Optimal
    }

    float
    compare_embeddings(const std::vector<float>& embedding1,
                       const std::vector<float>& embedding2)
    {
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
    get_max_distance() const
    {
        return max_distance_;
    }

    EnrollmentState
    enroll_face(const cv::Mat &frame)
    {
        std::vector<Face> faces = detect_faces(frame);
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

        std::vector<float> embedding = get_face_embedding(face_img);
        if (embedding.empty()) {
            g_debug("Failed to generate face embedding");
            return ENROLLMENT_FAIL;
        }

        int brightness = check_brightness(embedding);
        if (brightness < 1) {
            g_debug((brightness == -1) ? "Poor lighting conditions" : "Suboptimal lighting");
            return ENROLLMENT_BAD_LIGHTING;
        }

        pending_enrollments_.push_back(embedding);
        int count = pending_enrollments_.size();

        if (count >= 10) {
            std::vector<float> avg(embedding.size(), 0.0f);
            for (auto &emb : pending_enrollments_) {
                for (size_t i = 0; i < emb.size(); i++)
                    avg[i] += emb[i];
            }

            for (size_t i = 0; i < avg.size(); i++)
                avg[i] /= count;

            enrolled_embedding_ = avg;
            pending_enrollments_.clear();
            int save_status = save_enrolled_face();
            if (!save_status) {
                g_debug("Enrollment complete, but failed to save enrolled face");
                return ENROLLMENT_SAVE_FAILED;
            }

            g_debug("Enrollment complete");
            return ENROLLMENT_COMPLETE;
        } else {
            int progress = (count * 100) / 10;
            g_debug("Enrollment progress: %d%%", progress);
            return ENROLLMENT_IN_PROGRESS;
        }
    }

    RecognitionState
    recognize_face(const cv::Mat &frame)
    {
        std::vector<Face> faces = detect_faces(frame);
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
    is_enrolled() const
    {
        return !enrolled_embedding_.empty();
    }

private:
    std::unique_ptr<tflite::FlatBufferModel> detection_model_;
    std::unique_ptr<tflite::Interpreter> detector_;
    int detect_input_index_;
    int detect_output_boxes_;
    int detect_output_classes_;
    int detect_output_scores_;

    std::unique_ptr<tflite::FlatBufferModel> recognition_model_;
    std::unique_ptr<tflite::Interpreter> recognizer_;
    int recog_input_index_;
    int recog_output_index_;

    float min_confidence_;
    float max_distance_;

    std::vector<float> brightness_test_white_;
    std::vector<float> brightness_test_black_;

    std::mutex detector_mutex_;
    std::mutex recognizer_mutex_;

    std::vector<std::vector<float>> pending_enrollments_;
    std::vector<float> enrolled_embedding_;

    void
    create_brightness_test()
    {
        cv::Mat white_img(FR_INPUT_SIZE, FR_INPUT_SIZE, CV_8UC3, cv::Scalar(255, 255, 255));
        brightness_test_white_ = get_face_embedding(white_img);

        cv::Mat black_img(FR_INPUT_SIZE, FR_INPUT_SIZE, CV_8UC3, cv::Scalar(0, 0, 0));
        brightness_test_black_ = get_face_embedding(black_img);
    }

    std::vector<float>
    normalize_vector(const std::vector<float>& vec)
    {
        float norm = 0.0f;
        for (float val : vec) {
            norm += val * val;
        }

        norm = std::sqrt(norm);

        std::vector<float> normalized(vec.size());
        for (size_t i = 0; i < vec.size(); ++i) {
            normalized[i] = vec[i] / norm;
        }

        return normalized;
    }

    int
    save_enrolled_face()
    {
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
        } catch (const std::exception &e) {
            g_debug("Error saving enrollment: %s", e.what());
            return 0;
        }
    }

    int
    load_enrolled_face()
    {
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
            enrolled_embedding_ = j.get<std::vector<float>>();
            file.close();
            g_debug("Loaded enrolled face from %s", enrollment_file.string().c_str());
            return 1;
        } catch (const std::exception &e) {
            g_debug("Error loading enrolled face: %s", e.what());
            enrolled_embedding_.clear();
            return 0;
        }
    }

    fs::path
    get_data_dir()
    {
        const char *home = std::getenv("HOME");
        if (!home)
            throw std::runtime_error("Could not determine HOME directory");

        fs::path data_dir(home);
        data_dir /= ".local/share/faceauth";
        return data_dir;
    }
};

struct FacialUniversalRecognition {
    FaceDetector *instance;
};

extern "C" {

FacialUniversalRecognition*
face_create(const char *detection_model, const char *recognition_model)
{
    FacialUniversalRecognition *wrapper = new FacialUniversalRecognition;
    try {
        wrapper->instance = new FaceDetector(std::string(detection_model),
                                             std::string(recognition_model));
    } catch (const std::exception &e) {
        std::cerr << "Error in face_create: " << e.what() << std::endl;
        delete wrapper;
        return nullptr;
    }

    return wrapper;
}

void
face_destroy(FacialUniversalRecognition *handle)
{
    if (handle) {
        delete handle->instance;
        delete handle;
    }
}

EnrollmentState
face_enroll(FacialUniversalRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels)
{
    if (!handle || !handle->instance)
        return ENROLLMENT_FAIL;

    cv::Mat image(height, width, (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void*)image_data);
    EnrollmentState state = handle->instance->enroll_face(image);
    return state;
}

RecognitionState
face_recognize(FacialUniversalRecognition *handle,
               const unsigned char *image_data,
               int width, int height, int channels)
{
    if (!handle || !handle->instance)
        return RECOGNITION_FAIL;

    cv::Mat image(height, width, (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void*)image_data);
    RecognitionState state = handle->instance->recognize_face(image);
    return state;
}

int
face_is_enrolled(FacialUniversalRecognition *handle)
{
    if (!handle || !handle->instance)
        return 0;

    return handle->instance->is_enrolled() ? 1 : 0;
}

Face*
face_detect(FacialUniversalRecognition *handle,
            const unsigned char *image_data,
            int width, int height, int channels)
{
    if (!handle || !handle->instance)
        return nullptr;

    cv::Mat image(height, width,
                  (channels == 1) ? CV_8UC1 : CV_8UC(channels),
                  (void*)image_data);
    std::vector<FaceDetector::Face> detected = handle->instance->detect_faces(image);

    int count = detected.size();
    if (count == 0)
        return nullptr;

    Face* faces_array = (Face*)malloc(count * sizeof(Face));
    for (size_t i = 0; i < detected.size(); ++i) {
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
face_free_faces(Face *faces)
{
    if (faces)
        free(faces);
}

} // extern "C"
