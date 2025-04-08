#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gtk/gtk.h>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

class FaceDetectorRecognizer {
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

    FaceDetectorRecognizer(const std::string& detection_model_path,
                           const std::string& recognition_model_path,
                           float min_confidence = 0.5f,
                           float max_distance = 0.7f)
        : min_confidence_(min_confidence), max_distance_(max_distance)
        {

        g_debug("Initializing FaceDetectorRecognizer...");

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
    }

    std::vector<Face>
    detect_faces(const cv::Mat& image)
    {
        if (image.empty())
            return {};

        int height = image.rows;
        int width = image.cols;

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

            g_debug("Input tensor shape: %dx%dx%d", resized_image.cols,
                     resized_image.rows, resized_image.channels());

            int input_index = detector_->inputs()[0];
            TfLiteTensor* input_tensor = detector_->tensor(input_index);

            if (!input_tensor) {
                g_debug("Failed to get input tensor");
                return {};
            }

            g_debug("Input tensor found with type: %d", input_tensor->type);

            uint8_t* input_data = input_tensor->data.uint8;
            if (!input_data) {
                g_debug("Input tensor data is null");
                return {};
            }

            memcpy(input_data, resized_image.data, resized_image.total() * resized_image.elemSize());

            g_debug("Running inference...");
            if (detector_->Invoke() != kTfLiteOk) {
                g_debug("Failed to invoke detection model");
                return {};
            }

            int boxes_index = detector_->outputs()[0];
            int scores_index = detector_->outputs()[2];

            TfLiteTensor* boxes_tensor = detector_->tensor(boxes_index);
            TfLiteTensor* scores_tensor = detector_->tensor(scores_index);

            if (!boxes_tensor || !scores_tensor) {
                g_debug("Failed to get output tensors");
                return {};
            }

            const float* boxes_data = reinterpret_cast<const float*>(boxes_tensor->data.data);
            const float* scores_data = reinterpret_cast<const float*>(scores_tensor->data.data);

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

                    g_debug("Face detected: [%d, %d, %d, %d] score=%f", xmin_px,
                            ymin_px, xmax_px, ymax_px, scores_data[i]);

                    Face face;
                    face.bbox = cv::Rect(xmin_px, ymin_px, xmax_px - xmin_px, ymax_px - ymin_px);
                    face.confidence = scores_data[i];
                    detected_faces.push_back(face);
                }
            }

            g_debug("Detected %zu faces", detected_faces.size());
            return detected_faces;
        } catch (const std::exception& e) {
            g_debug("Exception in detect_faces: %s", e.what());
            return {};
        }
    }

    std::vector<float>
    get_face_embedding(const cv::Mat& face_image)
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
                uint8_t* input = recognizer_->typed_input_tensor<uint8_t>(recog_input_index_);
                if (!input)
                    return std::vector<float>();

                memcpy(input, processed_image.data,
                       processed_image.total() * processed_image.elemSize());
            } else {
                float* input = recognizer_->typed_input_tensor<float>(recog_input_index_);
                if (!input)
                    return std::vector<float>();

                cv::Mat float_image;
                processed_image.convertTo(float_image, CV_32F);
                float_image = (float_image - 127.5f) / 127.5f;

                memcpy(input, float_image.data,
                       float_image.total() * float_image.elemSize());
            }

            if (recognizer_->Invoke() != kTfLiteOk)
                return std::vector<float>();

            TfLiteTensor* output_tensor = recognizer_->tensor(recog_output_index_);
            if (!output_tensor)
                return std::vector<float>();

            const float* output_data = reinterpret_cast<const float*>(output_tensor->data.data);
            if (!output_data)
                return std::vector<float>();

            int embedding_size = output_tensor->dims->data[1];

            std::vector<float> embedding(output_data, output_data + embedding_size);
            return embedding;
        } catch (const std::exception&) {
            return std::vector<float>();
        }
    }

    cv::Mat
    extract_face(const cv::Mat& image, const cv::Rect& bbox)
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
};

class FaceAuthApp {
public:
    FaceAuthApp(const std::string& detection_model_path,
                const std::string& recognition_model_path,
                const std::string& mode = "enroll")
        : detector_(detection_model_path, recognition_model_path),
          mode_(mode),
          last_capture_time_(0),
          pipeline_(nullptr),
          appsink_(nullptr),
          g_main_loop_(nullptr),
          window_(nullptr),
          status_label_(nullptr)
        {

        g_debug("Initializing Face Authentication Application in %s mode", mode.c_str());

        data_dir_ = get_data_dir();
        fs::create_directories(data_dir_);
        enrollment_file_ = data_dir_ / "enrolled_face.json";

        if (mode_ == "recognize" && !fs::exists(enrollment_file_)) {
            g_debug("No enrollment found, switching to enrollment mode");
            mode_ = "enroll";
        }

        if (mode_ == "recognize")
            load_enrolled_face();
    }

    ~FaceAuthApp() {
        if (g_main_loop_ && g_main_loop_is_running(g_main_loop_))
            g_main_loop_quit(g_main_loop_);

        if (window_) {
            gtk_widget_destroy(window_);
            window_ = nullptr;
        }

        if (pipeline_) {
            gst_element_set_state(pipeline_, GST_STATE_NULL);
            gst_object_unref(pipeline_);
            pipeline_ = nullptr;
        }

        if (appsink_) {
            gst_object_unref(appsink_);
            appsink_ = nullptr;
        }
    }

    bool
    setup_pipeline()
    {
        try {
            const std::string pipeline_str =
                "droidcamsrc camera_device=1 mode=2 ! tee name=t "
                "t. ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, width=640, height=480 ! videoconvert ! "
                "videoflip video-direction=auto ! gtksink name=sink sync=false "
                "t. ! queue max-size-buffers=1 leaky=downstream ! videoconvert ! video/x-raw, format=RGB ! "
                "videoflip video-direction=auto ! appsink name=appsink max-buffers=1 drop=true emit-signals=true sync=false";

            g_debug("Creating GStreamer pipeline");
            GError *error = nullptr;
            pipeline_ = gst_parse_launch(pipeline_str.c_str(), &error);

            if (error) {
                g_debug("Failed to create pipeline: %s", error->message);
                g_error_free(error);
                return false;
            }

            appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "appsink");
            if (!appsink_) {
                g_debug("Failed to get appsink element");
                return false;
            }

            GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
            if (!sink) {
                g_debug("Failed to get sink element");
                return false;
            }

            window_ = gtk_window_new(GTK_WINDOW_TOPLEVEL);
            gtk_window_set_title(GTK_WINDOW(window_), "Face Authentication");
            gtk_window_set_default_size(GTK_WINDOW(window_), 640, 480);
            g_signal_connect(window_, "destroy", G_CALLBACK(+[](GtkWidget* widget, gpointer data) {
                auto app = static_cast<FaceAuthApp*>(data);
                gtk_main_quit();
            }), this);

            GtkWidget* video_widget = nullptr;
            g_object_get(G_OBJECT(sink), "widget", &video_widget, NULL);
            if (video_widget)
                gtk_container_add(GTK_CONTAINER(window_), video_widget);
            else
                g_debug("Failed to get video widget from sink");

            gtk_widget_show_all(window_);

            g_object_set(G_OBJECT(appsink_), "emit-signals", TRUE, NULL);

            g_signal_connect(appsink_, "new-sample", G_CALLBACK(+[](GstElement* appsink, gpointer user_data) -> GstFlowReturn {
                return static_cast<FaceAuthApp*>(user_data)->on_new_sample(appsink);
            }), this);

            GstBus *bus = gst_element_get_bus(pipeline_);
            gst_bus_add_watch(bus, (GstBusFunc)(+[](GstBus* bus, GstMessage* msg, gpointer user_data) -> gboolean {
                return static_cast<FaceAuthApp*>(user_data)->on_bus_message(bus, msg);
            }), this);
            gst_object_unref(bus);

            GstStateChangeReturn ret = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
            if (ret == GST_STATE_CHANGE_FAILURE) {
                g_debug("Failed to start pipeline");
                return false;
            }

            g_debug("Pipeline started successfully");
            return true;
        } catch (const std::exception& e) {
            g_debug("Error setting up pipeline: %s", e.what());
            return false;
        }
    }

    void
    run()
    {
        if (!setup_pipeline())
            throw std::runtime_error("Failed to setup GStreamer pipeline");

        g_debug("Starting main loop");

        if (!gtk_init_check(nullptr, nullptr)) {
            g_debug("Failed to initialize GTK");
            throw std::runtime_error("Failed to initialize GTK");
        }

        status_label_ = gtk_label_new("Initializing...");
        gtk_widget_set_name(status_label_, "status_label");

        GtkCssProvider* provider = gtk_css_provider_new();
        gtk_css_provider_load_from_data(provider,
            "#status_label {"
            "   color: white;"
            "   font-weight: bold;"
            "   background-color: rgba(0,0,0,0.5);"
            "   padding: 5px;"
            "   border-radius: 5px;"
            "}", -1, nullptr);

        GtkStyleContext* context = gtk_widget_get_style_context(status_label_);
        gtk_style_context_add_provider(context, GTK_STYLE_PROVIDER(provider), GTK_STYLE_PROVIDER_PRIORITY_USER);

        GtkFixed* fixed = GTK_FIXED(gtk_fixed_new());
        gtk_fixed_put(fixed, status_label_, 10, 10);
        gtk_widget_show(status_label_);
        gtk_widget_show(GTK_WIDGET(fixed));

        gtk_container_add(GTK_CONTAINER(window_), GTK_WIDGET(fixed));
        gtk_widget_show_all(window_);

        g_main_loop_ = g_main_loop_new(nullptr, FALSE);

        gtk_main();

        if (g_main_loop_) {
            g_main_loop_unref(g_main_loop_);
            g_main_loop_ = nullptr;
        }

        g_debug("Main loop exited");
    }

    GstFlowReturn
    on_new_sample(GstElement* appsink)
    {
        static auto last_process_time = std::chrono::steady_clock::now();

        auto current_time = std::chrono::steady_clock::now();

        auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_process_time).count();

        if (time_diff < 500) {
            GstSample* sample = nullptr;
            g_signal_emit_by_name(appsink, "pull-sample", &sample);
            if (sample)
                gst_sample_unref(sample);

            return GST_FLOW_OK;
        }

        last_process_time = current_time;

        GstSample* sample = nullptr;
        g_signal_emit_by_name(appsink, "pull-sample", &sample);

        if (!sample)
            return GST_FLOW_ERROR;

        try {
            GstBuffer* buffer = gst_sample_get_buffer(sample);
            GstCaps* caps = gst_sample_get_caps(sample);
            GstStructure* structure = gst_caps_get_structure(caps, 0);

            int width, height;
            gst_structure_get_int(structure, "width", &width);
            gst_structure_get_int(structure, "height", &height);

            GstMapInfo map;
            if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
                cv::Mat temp(height, width, CV_8UC3, (void*)map.data);
                cv::Mat frame = temp.clone();

                gst_buffer_unmap(buffer, &map);
                if (!frame.empty())
                    process_frame(frame);
            }
        } catch (const std::exception& e) {
            g_debug("Error processing frame: %s", e.what());
        }

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    bool
    on_bus_message(GstBus* bus, GstMessage* message)
    {
        switch (GST_MESSAGE_TYPE(message)) {
            case GST_MESSAGE_ERROR: {
                GError* err = nullptr;
                gchar* debug_info = nullptr;
                gst_message_parse_error(message, &err, &debug_info);

                g_debug("Pipeline error: %s", err->message);
                g_debug("Debug info: %s", debug_info ? debug_info : "none");

                g_clear_error(&err);
                g_free(debug_info);

                g_main_loop_quit(g_main_loop_);
                return FALSE;
            }
            case GST_MESSAGE_WARNING: {
                GError* err = nullptr;
                gchar* debug_info = nullptr;
                gst_message_parse_warning(message, &err, &debug_info);

                g_debug("Pipeline warning: %s", err->message);

                g_clear_error(&err);
                g_free(debug_info);
                break;
            }
            case GST_MESSAGE_EOS:
                g_debug("End of stream");
                g_main_loop_quit(g_main_loop_);
                break;
            case GST_MESSAGE_STATE_CHANGED: {
                if (GST_MESSAGE_SRC(message) == GST_OBJECT(pipeline_)) {
                    GstState old_state, new_state, pending_state;
                    gst_message_parse_state_changed(message, &old_state, &new_state, &pending_state);

                    g_debug("Pipeline state changed: %s -> %s",
                            gst_element_state_get_name(old_state),
                            gst_element_state_get_name(new_state));
                }
                break;
            }
            default:
                break;
        }
        return TRUE;
    }

    void
    update_status(const std::string& text)
    {
        if (status_label_) {
            g_idle_add(+[](gpointer user_data) -> gboolean {
                auto data = static_cast<std::pair<GtkWidget*, std::string>*>(user_data);
                gtk_label_set_text(GTK_LABEL(data->first), data->second.c_str());
                delete data;
                return FALSE;
            }, new std::pair<GtkWidget*, std::string>(status_label_, text));
        }
    }

    void
    process_frame(cv::Mat& frame)
    {
        try {
            if (frame.empty())
                return;

            cv::Mat process_frame = frame.clone();

            std::vector<FaceDetectorRecognizer::Face> faces = detector_.detect_faces(process_frame);

            std::string status_text;
            if (mode_ == "enroll")
                status_text = handle_enrollment(process_frame, faces);
            else
                status_text = handle_recognition(process_frame, faces);

            update_status(status_text);

            static int frame_count = 0;
            if (++frame_count % 30 == 0) {
                g_debug("Status: %s", status_text.c_str());
                g_debug("Detected faces: %zu", faces.size());
            }
        } catch (const std::exception& e) {
            g_debug("Error in process_frame: %s", e.what());
        }
    }

    std::string
    handle_enrollment(const cv::Mat& frame,
                      const std::vector<FaceDetectorRecognizer::Face>& faces)
    {
        try {
            if (embeddings_.size() >= 10) {
                save_enrolled_face();
                mode_ = "recognize";
                load_enrolled_face();
                return "Enrollment complete! Switched to recognition mode";
            }

            if (faces.size() == 1) {
                double current_time = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
                if (current_time - last_capture_time_ > 0.5) {
                    auto face = faces[0];

                    cv::Mat face_img = detector_.extract_face(frame, face.bbox);

                    if (!face_img.empty()) {
                        std::vector<float> embedding = detector_.get_face_embedding(face_img);

                        if (embedding.empty())
                            return "Failed to generate face embedding";

                        int brightness = detector_.check_brightness(embedding);

                        if (brightness >= 1) {
                            embeddings_.push_back(embedding);
                            last_capture_time_ = current_time;
                            int progress = embeddings_.size() * 10;

                            g_debug("Capture %zu/10 successful", embeddings_.size());

                            return "Enrollment Progress: " + std::to_string(progress) + "%";
                        } else {
                            return brightness == -1 ? "Poor lighting conditions" : "Suboptimal lighting";
                        }
                    } else {
                        return "Invalid face image extracted";
                    }
                }
                return "Hold still...";
            } else {
                return faces.size() > 1 ? "Multiple faces detected" : "No face detected";
            }
        /* none of these will ever be reached but i really need something that doesn't segfault */
        } catch (const std::exception& e) {
            g_debug("Error in handle_enrollment: %s", e.what());
            return "Error processing face";
        }
    }

    std::string
    handle_recognition(const cv::Mat& frame,
                       const std::vector<FaceDetectorRecognizer::Face>& faces)
    {
        try {
            if (faces.size() == 1) {
                auto face = faces[0];

                cv::Mat face_img = detector_.extract_face(frame, face.bbox);

                if (!face_img.empty()) {
                    std::vector<float> embedding = detector_.get_face_embedding(face_img);

                    if (embedding.empty())
                        return "Failed to generate face embedding";

                    if (enrolled_embeddings_.empty())
                        return "No enrolled faces found";

                    float min_distance = std::numeric_limits<float>::max();
                    int best_match_index = -1;

                    for (size_t i = 0; i < enrolled_embeddings_.size(); i++) {
                        float distance = detector_.compare_embeddings(embedding, enrolled_embeddings_[i]);
                        if (distance < min_distance) {
                            min_distance = distance;
                            best_match_index = i;
                        }
                    }

                    bool recognized = min_distance < detector_.get_max_distance();
                    std::stringstream ss;

                    if (recognized)
                        ss << "Recognized Enrolled Face #" << (best_match_index + 1)
                           << " (Distance: " << std::fixed << std::setprecision(3) << min_distance << ")";
                    else
                        ss << "Not Recognized - Unknown Face"
                           << " (Distance: " << std::fixed << std::setprecision(3) << min_distance << ")";

                    g_debug("Recognition result: %s", ss.str().c_str());
                    return ss.str();
                } else {
                    return "Invalid face image extracted";
                }
            } else {
                return faces.size() > 1 ? "Multiple faces detected" : "No face detected";
            }
        } catch (const std::exception& e) {
            g_debug("Error in handle_recognition: %s", e.what());
            return "Error processing face";
        }
    }

    void
    save_enrolled_face()
    {
        try {
            json j = embeddings_;
            std::ofstream file(enrollment_file_);
            file << j;
            g_debug("Enrollment saved to %s", enrollment_file_.string().c_str());
        } catch (const std::exception& e) {
            g_debug("Error saving enrollment: %s", e.what());
        }
    }

    void
    load_enrolled_face()
    {
        try {
            std::ifstream file(enrollment_file_);
            json j;
            file >> j;
            enrolled_embeddings_ = j.get<std::vector<std::vector<float>>>();

            g_debug("Loaded %zu enrolled face samples", enrolled_embeddings_.size());
        } catch (const std::exception& e) {
            g_debug("Error loading enrolled face: %s", e.what());
            enrolled_embeddings_.clear();
        }
    }

    fs::path
    get_data_dir()
    {
        const char* home_dir = std::getenv("HOME");
        if (!home_dir)
            throw std::runtime_error("Could not determine home directory");

        return fs::path(home_dir) / ".local" / "share" / "faceauth";
    }

private:
    FaceDetectorRecognizer detector_;
    std::string mode_;
    std::vector<std::vector<float>> embeddings_;
    std::vector<std::vector<float>> enrolled_embeddings_;
    double last_capture_time_;
    fs::path data_dir_;
    fs::path enrollment_file_;

    GstElement* pipeline_;
    GstElement* appsink_;
    GMainLoop* g_main_loop_;

    GtkWidget* window_;
    GtkWidget* status_label_;
};

int
main(int argc, char* argv[])
{
    g_debug("Starting Face Authentication System");

    gtk_init(&argc, &argv);
    gst_init(&argc, &argv);

    try {
        std::string detection_model = "detect-class1.tflite";
        std::string recognition_model = "mobile_face_net.tflite";

        if (argc > 1)
            detection_model = argv[1];
        if (argc > 2)
            recognition_model = argv[2];

        if (!fs::exists(detection_model)) {
            g_debug("Error: Detection model not found: %s", detection_model.c_str());
            return 1;
        }

        if (!fs::exists(recognition_model)) {
            g_debug("Error: Recognition model not found: %s", recognition_model.c_str());
            return 1;
        }

        fs::path home_dir = std::getenv("HOME");
        fs::path enrollment_file = home_dir / ".local" / "share" / "faceauth" / "enrolled_face.json";

        std::string initial_mode;
        if (fs::exists(enrollment_file)) {
            g_debug("Found existing face enrollment");
            initial_mode = "recognize";
        } else {
            g_debug("No enrollment found - starting enrollment process");
            initial_mode = "enroll";
        }

        g_debug("Creating FaceAuthApp instance...");
        FaceAuthApp app(detection_model, recognition_model, initial_mode);

        g_debug("Starting application main loop...");
        app.run();
    } catch (const std::exception& e) {
        g_debug("Error: %s", e.what());
        return 1;
    }

    g_debug("Application terminated normally");
    return 0;
}
