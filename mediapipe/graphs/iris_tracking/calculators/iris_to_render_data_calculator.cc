// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <memory>

#include "absl/strings/str_cat.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/graphs/iris_tracking/calculators/iris_to_render_data_calculator.pb.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"

namespace mediapipe {

namespace {

constexpr char kIrisTag[] = "IRIS";
constexpr char kRenderDataTag[] = "RENDER_DATA";
constexpr char kImageSizeTag[] = "IMAGE_SIZE";
constexpr char kLeftIrisDepthTag[] = "LEFT_IRIS_DEPTH_MM";
constexpr char kRightIrisDepthTag[] = "RIGHT_IRIS_DEPTH_MM";
constexpr char kFaceLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kOvalLabel[] = "OVAL";
constexpr float kFontHeightScale = 1.5f;
constexpr int kNumIrisLandmarksPerEye = 5;
// TODO: Source.
constexpr float kIrisSizeInMM = 11.8;
constexpr float kDeltaAdjustInMM = 4;
constexpr float kDeltaStrabismusThresholdInMM = 6;
constexpr float kDepthWeightUpdate = 0.1;

inline void SetColor(RenderAnnotation* annotation, const Color& color) {
  annotation->mutable_color()->set_r(color.r());
  annotation->mutable_color()->set_g(color.g());
  annotation->mutable_color()->set_b(color.b());
}

inline float GetDepth(float x0, float y0, float x1, float y1) {
  return std::sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

inline float GetLandmarkDepth(const NormalizedLandmark& ld0,
                              const NormalizedLandmark& ld1,
                              const std::pair<int, int>& image_size) {
  return GetDepth(ld0.x() * image_size.first, ld0.y() * image_size.second,
                  ld1.x() * image_size.first, ld1.y() * image_size.second);
}

float CalculateIrisDiameter(const NormalizedLandmarkList& landmarks,
                            const std::pair<int, int>& image_size) {
  const float dist_vert = GetLandmarkDepth(landmarks.landmark(1),
                                           landmarks.landmark(2), image_size);
  const float dist_hori = GetLandmarkDepth(landmarks.landmark(3),
                                           landmarks.landmark(4), image_size);
  return (dist_hori + dist_vert) / 2.0f;
}

float CalculateDepth(const NormalizedLandmark& center, float focal_length,
                     float iris_size, float img_w, float img_h) {
  std::pair<float, float> origin{img_w / 2.f, img_h / 2.f};
  const auto y = GetDepth(origin.first, origin.second, center.x() * img_w,
                          center.y() * img_h);
  const auto x = std::sqrt(focal_length * focal_length + y * y);
  const auto depth = kIrisSizeInMM * x / iris_size;
  return depth;
}

}  // namespace

// Converts iris landmarks to render data and estimates depth from the camera if
// focal length and image size. The depth will be rendered as part of the render
// data on the frame.
//
// Usage example:
// node {
//   calculator: "IrisToRenderDataCalculator"
//   input_stream: "IRIS:iris_landmarks"
//   input_stream: "IMAGE_SIZE:image_size"
//   # Note: Only one of FOCAL_LENGTH or IMAGE_FILE_PROPERTIES is necessary
//   # to get focal length in pixels. Sending focal length in pixels to
//   # this calculator is optional.
//   input_side_packet: "FOCAL_LENGTH:focal_length_pixel"
//   # OR
//   input_side_packet: "IMAGE_FILE_PROPERTIES:image_file_properties"
//   output_stream: "RENDER_DATA:iris_render_data"
//   output_stream: "LEFT_IRIS_DEPTH_MM:left_iris_depth_mm"
//   output_stream: "RIGHT_IRIS_DEPTH_MM:right_iris_depth_mm"
//   node_options: {
//     [type.googleapis.com/mediapipe.IrisToRenderDataCalculatorOptions] {
//       color { r: 255 g: 255 b: 255 }
//       thickness: 2.0
//       font_height_px: 50
//       horizontal_offset_px: 200
//       vertical_offset_px: 200
//       location: TOP_LEFT
//     }
//   }
// }
class IrisToRenderDataCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kIrisTag).Set<NormalizedLandmarkList>();
    cc->Inputs().Tag(kFaceLandmarksTag).Set<NormalizedLandmarkList>();
    cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
    cc->Inputs().Tag(kImageSizeTag).Set<std::pair<int, int>>();

    if (cc->Inputs().HasTag(kLeftIrisDepthTag)) {
      cc->Inputs().Tag(kLeftIrisDepthTag).Set<float>();
    }
    if (cc->Inputs().HasTag(kRightIrisDepthTag)) {
      cc->Inputs().Tag(kRightIrisDepthTag).Set<float>();
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

 private:

  float last_plu_dt_a_r_x = -1.f;
  float last_plu_dt_a_r_y = -1.f;
  float last_plu_dt_a_r = -1.f;
  
  float last_plu_dt_a_l_x = -1.f;
  float last_plu_dt_a_l_y = -1.f;
  float last_plu_dt_a_l = -1.f;
  
  float last_plu_dt_t_r_x = -1.f;
  float last_plu_dt_t_r_y = -1.f;
  float last_plu_dt_t_r = -1.f;
  
  float last_plu_dt_t_l_x = -1.f;
  float last_plu_dt_t_l_y = -1.f;
  float last_plu_dt_t_l = -1.f;
  
  float last_plu_dt_n_r_x = -1.f;
  float last_plu_dt_n_r_y = -1.f;
  float last_plu_dt_n_r = -1.f;
  
  float last_plu_dt_n_l_x = -1.f;
  float last_plu_dt_n_l_y = -1.f;
  float last_plu_dt_n_l = -1.f;

  int warn_delta_plu_r_x_count = 0;
  int warn_delta_plu_l_x_count = 0;
  int warn_delta_plu_r_y_count = 0;
  int warn_delta_plu_l_y_count = 0;
  int warn_delta_plu_r_count = 0;
  int warn_delta_plu_l_count = 0;

  int warn_delta_plu_x_count = 0;
  int warn_delta_plu_y_count = 0;
  int warn_delta_plu_count = 0;
  
  float plu_iris_size = -1.f;

  float plu_dt_a_r_x = -1.f;
  float plu_dt_a_r_y = -1.f;
  float plu_dt_a_r = -1.f;
  
  float plu_dt_a_l_x = -1.f;
  float plu_dt_a_l_y = -1.f;
  float plu_dt_a_l = -1.f;
  
  float plu_dt_t_r_x = -1.f;
  float plu_dt_t_r_y = -1.f;
  float plu_dt_t_r = -1.f;
  
  float plu_dt_t_l_x = -1.f;
  float plu_dt_t_l_y = -1.f;
  float plu_dt_t_l = -1.f;
  
  float plu_dt_n_r_x = -1.f;
  float plu_dt_n_r_y = -1.f;
  float plu_dt_n_r = -1.f;
  
  float plu_dt_n_l_x = -1.f;
  float plu_dt_n_l_y = -1.f;
  float plu_dt_n_l = -1.f;


  void RenderIris(const NormalizedLandmarkList& iris_landmarks,
                  const IrisToRenderDataCalculatorOptions& options,
                  const std::pair<int, int>& image_size, float iris_size,
                  RenderData* render_data);
  void GetLeftIris(const NormalizedLandmarkList& lds,
                   NormalizedLandmarkList* iris);
  void GetRightIris(const NormalizedLandmarkList& lds,
                    NormalizedLandmarkList* iris);

  void AddTextRenderData(const IrisToRenderDataCalculatorOptions& options,
                         const std::pair<int, int>& image_size,
                         const std::vector<std::string>& lines,
                         RenderData* render_data);

  static RenderAnnotation* AddOvalRenderData(
      const IrisToRenderDataCalculatorOptions& options,
      RenderData* render_data);
  static RenderAnnotation* AddPointRenderData(
      const IrisToRenderDataCalculatorOptions& options,
      RenderData* render_data);
};
REGISTER_CALCULATOR(IrisToRenderDataCalculator);

absl::Status IrisToRenderDataCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return absl::OkStatus();
}

absl::Status IrisToRenderDataCalculator::Process(CalculatorContext* cc) {
  // Only process if there's input landmarks.
  if (cc->Inputs().Tag(kIrisTag).IsEmpty()) {
    return absl::OkStatus();
  }
  const auto& options =
      cc->Options<::mediapipe::IrisToRenderDataCalculatorOptions>();

  const auto& iris_landmarks =
      cc->Inputs().Tag(kIrisTag).Get<NormalizedLandmarkList>();
  RET_CHECK_EQ(iris_landmarks.landmark_size(), kNumIrisLandmarksPerEye * 2)
      << "Wrong number of iris landmarks";

  std::pair<int, int> image_size;
  RET_CHECK(!cc->Inputs().Tag(kImageSizeTag).IsEmpty());
  image_size = cc->Inputs().Tag(kImageSizeTag).Get<std::pair<int, int>>();

  auto render_data = absl::make_unique<RenderData>();
  auto left_iris = absl::make_unique<NormalizedLandmarkList>();
  auto right_iris = absl::make_unique<NormalizedLandmarkList>();
  GetLeftIris(iris_landmarks, left_iris.get());
  GetRightIris(iris_landmarks, right_iris.get());

  const auto left_iris_size = CalculateIrisDiameter(*left_iris, image_size);
  const auto right_iris_size = CalculateIrisDiameter(*right_iris, image_size);
  RenderIris(*left_iris, options, image_size, left_iris_size,
             render_data.get());
  RenderIris(*right_iris, options, image_size, right_iris_size,
             render_data.get());

  std::vector<std::string> lines;
  std::string line;
  if (cc->Inputs().HasTag(kLeftIrisDepthTag) &&
      !cc->Inputs().Tag(kLeftIrisDepthTag).IsEmpty()) {
    const float left_iris_depth =
        cc->Inputs().Tag(kLeftIrisDepthTag).Get<float>();
    if (!std::isinf(left_iris_depth)) {
      line = "Left : ";
      absl::StrAppend(&line, ":", std::round(left_iris_depth / 10), " cm");
      lines.emplace_back(line);
    }
  }
  if (cc->Inputs().HasTag(kRightIrisDepthTag) &&
      !cc->Inputs().Tag(kRightIrisDepthTag).IsEmpty()) {
    const float right_iris_depth =
        cc->Inputs().Tag(kRightIrisDepthTag).Get<float>();
    if (!std::isinf(right_iris_depth)) {
      line = "Right : ";
      absl::StrAppend(&line, ":", std::round(right_iris_depth / 10), " cm");
      lines.emplace_back(line);
    }
  }

  if (!cc->Inputs().Tag(kIrisTag).IsEmpty()) {
    const auto& plu_c_r = right_iris->landmark(0);
    const auto& plu_c_l = left_iris->landmark(0);
    // get 4 eye points
    if (!cc->Inputs().Tag(kFaceLandmarksTag).IsEmpty()) {
      const auto& update_face_landmarks =
          cc->Inputs().Tag(kFaceLandmarksTag).Get<NormalizedLandmarkList>();
      const auto& plu_t_r = update_face_landmarks.landmark(263);
      const auto& plu_n_r = update_face_landmarks.landmark(362);
      const auto& plu_n_l = update_face_landmarks.landmark(133);
      const auto& plu_t_l = update_face_landmarks.landmark(33);

      // draw 4 eye points
      auto* landmark_data_render = AddPointRenderData(options, render_data.get());
      auto* landmark_data = landmark_data_render->mutable_point();
      landmark_data->set_normalized(true);
      landmark_data->set_x(plu_t_r.x());
      landmark_data->set_y(plu_t_r.y());
      landmark_data_render = AddPointRenderData(options, render_data.get());
      landmark_data = landmark_data_render->mutable_point();
      landmark_data->set_normalized(true);
      landmark_data->set_x(plu_n_r.x());
      landmark_data->set_y(plu_n_r.y());
      landmark_data_render = AddPointRenderData(options, render_data.get());
      landmark_data = landmark_data_render->mutable_point();
      landmark_data->set_normalized(true);
      landmark_data->set_x(plu_n_l.x());
      landmark_data->set_y(plu_n_l.y());
      landmark_data_render = AddPointRenderData(options, render_data.get());
      landmark_data = landmark_data_render->mutable_point();
      landmark_data->set_normalized(true);
      landmark_data->set_x(plu_t_l.x());
      landmark_data->set_y(plu_t_l.y());

      // iris plu size
      auto raw_plu_iris_size = left_iris_size;
      if (raw_plu_iris_size < right_iris_size) {
        raw_plu_iris_size = right_iris_size;
      }
      if (raw_plu_iris_size > 0) {
        plu_iris_size =
            plu_iris_size < 0 || std::isinf(plu_iris_size)
                ? raw_plu_iris_size
                : plu_iris_size * (1 - kDepthWeightUpdate) +
                      raw_plu_iris_size * kDepthWeightUpdate;

        const auto plu_adjust_iris_size_ratio = kIrisSizeInMM / plu_iris_size;
        const auto show_plu_iris_size = plu_iris_size * plu_adjust_iris_size_ratio;
        const auto plu_left_iris_size = left_iris_size * plu_adjust_iris_size_ratio;
        const auto plu_right_iris_size = right_iris_size * plu_adjust_iris_size_ratio;

        const auto image_size_x = image_size.first * plu_adjust_iris_size_ratio;
        const auto image_size_y = image_size.second * plu_adjust_iris_size_ratio;

        const auto raw_plu_dt_a_r_x = std::sqrt((plu_t_r.x() - plu_n_r.x()) * (plu_t_r.x() - plu_n_r.x())) * image_size_x;
        const auto raw_plu_dt_a_r_y = std::sqrt((plu_t_r.y() - plu_n_r.y()) * (plu_t_r.y() - plu_n_r.y())) * image_size_y;
        const auto raw_plu_dt_a_r = std::sqrt(raw_plu_dt_a_r_x * raw_plu_dt_a_r_x + raw_plu_dt_a_r_y * raw_plu_dt_a_r_y);
        
        const auto raw_plu_dt_a_l_x = std::sqrt((plu_t_l.x() - plu_n_l.x()) * (plu_t_l.x() - plu_n_l.x())) * image_size_x;
        const auto raw_plu_dt_a_l_y = std::sqrt((plu_t_l.y() - plu_n_l.y()) * (plu_t_l.y() - plu_n_l.y())) * image_size_y;
        const auto raw_plu_dt_a_l = std::sqrt(raw_plu_dt_a_l_x * raw_plu_dt_a_l_x + raw_plu_dt_a_l_y * raw_plu_dt_a_l_y);
        
        const auto raw_plu_dt_t_r_x = std::sqrt((plu_t_r.x() - plu_c_r.x()) * (plu_t_r.x() - plu_c_r.x())) * image_size_x;
        const auto raw_plu_dt_t_r_y = std::sqrt((plu_t_r.y() - plu_c_r.y()) * (plu_t_r.y() - plu_c_r.y())) * image_size_y;
        const auto raw_plu_dt_t_r = std::sqrt(raw_plu_dt_t_r_x * raw_plu_dt_t_r_x + raw_plu_dt_t_r_y * raw_plu_dt_t_r_y);
        
        const auto raw_plu_dt_t_l_x = std::sqrt((plu_t_l.x() - plu_c_l.x()) * (plu_t_l.x() - plu_c_l.x())) * image_size_x;
        const auto raw_plu_dt_t_l_y = std::sqrt((plu_t_l.y() - plu_c_l.y()) * (plu_t_l.y() - plu_c_l.y())) * image_size_y;
        const auto raw_plu_dt_t_l = std::sqrt(raw_plu_dt_t_l_x * raw_plu_dt_t_l_x + raw_plu_dt_t_l_y * raw_plu_dt_t_l_y);
        
        const auto raw_plu_dt_n_r_x = std::sqrt((plu_n_r.x() - plu_c_r.x()) * (plu_n_r.x() - plu_c_r.x())) * image_size_x;
        const auto raw_plu_dt_n_r_y = std::sqrt((plu_n_r.y() - plu_c_r.y()) * (plu_n_r.y() - plu_c_r.y())) * image_size_y;
        const auto raw_plu_dt_n_r = std::sqrt(raw_plu_dt_n_r_x * raw_plu_dt_n_r_x + raw_plu_dt_n_r_y * raw_plu_dt_n_r_y);
        
        const auto raw_plu_dt_n_l_x = std::sqrt((plu_n_l.x() - plu_c_l.x()) * (plu_n_l.x() - plu_c_l.x())) * image_size_x;
        const auto raw_plu_dt_n_l_y = std::sqrt((plu_n_l.y() - plu_c_l.y()) * (plu_n_l.y() - plu_c_l.y())) * image_size_y;
        const auto raw_plu_dt_n_l = std::sqrt(raw_plu_dt_n_l_x * raw_plu_dt_n_l_x + raw_plu_dt_n_l_y * raw_plu_dt_n_l_y);


        plu_dt_a_r_x =
            plu_dt_a_r_x < 0 || std::isinf(plu_dt_a_r_x)
                ? raw_plu_dt_a_r_x
                : plu_dt_a_r_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_r_x * kDepthWeightUpdate;

        plu_dt_a_r_y =
            plu_dt_a_r_y < 0 || std::isinf(plu_dt_a_r_y)
                ? raw_plu_dt_a_r_y
                : plu_dt_a_r_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_r_y * kDepthWeightUpdate;

        plu_dt_a_r =
            plu_dt_a_r < 0 || std::isinf(plu_dt_a_r)
                ? raw_plu_dt_a_r
                : plu_dt_a_r * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_r * kDepthWeightUpdate;

        plu_dt_a_l_x =
            plu_dt_a_l_x < 0 || std::isinf(plu_dt_a_l_x)
                ? raw_plu_dt_a_l_x
                : plu_dt_a_l_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_l_x * kDepthWeightUpdate;

        plu_dt_a_l_y =
            plu_dt_a_l_y < 0 || std::isinf(plu_dt_a_l_y)
                ? raw_plu_dt_a_l_y
                : plu_dt_a_l_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_l_y * kDepthWeightUpdate;

        plu_dt_a_l =
            plu_dt_a_l < 0 || std::isinf(plu_dt_a_l)
                ? raw_plu_dt_a_l
                : plu_dt_a_l * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_a_l * kDepthWeightUpdate;

        plu_dt_t_r_x =
            plu_dt_t_r_x < 0 || std::isinf(plu_dt_t_r_x)
                ? raw_plu_dt_t_r_x
                : plu_dt_t_r_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_r_x * kDepthWeightUpdate;

        plu_dt_t_r_y =
            plu_dt_t_r_y < 0 || std::isinf(plu_dt_t_r_y)
                ? raw_plu_dt_t_r_y
                : plu_dt_t_r_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_r_y * kDepthWeightUpdate;

        plu_dt_t_r =
            plu_dt_t_r < 0 || std::isinf(plu_dt_t_r)
                ? raw_plu_dt_t_r
                : plu_dt_t_r * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_r * kDepthWeightUpdate;

        plu_dt_t_l_x =
            plu_dt_t_l_x < 0 || std::isinf(plu_dt_t_l_x)
                ? raw_plu_dt_t_l_x
                : plu_dt_t_l_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_l_x * kDepthWeightUpdate;

        plu_dt_t_l_y =
            plu_dt_t_l_y < 0 || std::isinf(plu_dt_t_l_y)
                ? raw_plu_dt_t_l_y
                : plu_dt_t_l_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_l_y * kDepthWeightUpdate;

        plu_dt_t_l =
            plu_dt_t_l < 0 || std::isinf(plu_dt_t_l)
                ? raw_plu_dt_t_l
                : plu_dt_t_l * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_t_l * kDepthWeightUpdate;

        plu_dt_n_r_x =
            plu_dt_n_r_x < 0 || std::isinf(plu_dt_n_r_x)
                ? raw_plu_dt_n_r_x
                : plu_dt_n_r_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_r_x * kDepthWeightUpdate;

        plu_dt_n_r_y =
            plu_dt_n_r_y < 0 || std::isinf(plu_dt_n_r_y)
                ? raw_plu_dt_n_r_y
                : plu_dt_n_r_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_r_y * kDepthWeightUpdate;

        plu_dt_n_r =
            plu_dt_n_r < 0 || std::isinf(plu_dt_n_r)
                ? raw_plu_dt_n_r
                : plu_dt_n_r * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_r * kDepthWeightUpdate;

        plu_dt_n_l_x =
            plu_dt_n_l_x < 0 || std::isinf(plu_dt_n_l_x)
                ? raw_plu_dt_n_l_x
                : plu_dt_n_l_x * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_l_x * kDepthWeightUpdate;

        plu_dt_n_l_y =
            plu_dt_n_l_y < 0 || std::isinf(plu_dt_n_l_y)
                ? raw_plu_dt_n_l_y
                : plu_dt_n_l_y * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_l_y * kDepthWeightUpdate;

        plu_dt_n_l =
            plu_dt_n_l < 0 || std::isinf(plu_dt_n_l)
                ? raw_plu_dt_n_l
                : plu_dt_n_l * (1 - kDepthWeightUpdate) +
                      raw_plu_dt_n_l * kDepthWeightUpdate;
        
        // calculate delta_xxx
        float delta_plu_r_x = -100;
        float delta_plu_l_x = -100;
        float delta_plu_r_y = -100;
        float delta_plu_l_y = -100;
        float delta_plu_r = -100;
        float delta_plu_l = -100;
        float delta_plu_x = -100;
        float delta_plu_y = -100;
        float delta_plu = -100;

        if (last_plu_dt_a_r > 0 && plu_dt_a_r > 0) {
          delta_plu_r_x = ((plu_dt_n_r_x - last_plu_dt_n_r_x) - (plu_dt_t_r_x - last_plu_dt_t_r_x)) * kDeltaAdjustInMM / 2;
          delta_plu_l_x = ((plu_dt_n_l_x - last_plu_dt_n_l_x) - (plu_dt_t_l_x - last_plu_dt_t_l_x)) * kDeltaAdjustInMM / 2;
          delta_plu_r_y = ((plu_dt_n_r_y - last_plu_dt_n_r_y) - (plu_dt_t_r_y - last_plu_dt_t_r_y)) * kDeltaAdjustInMM / 2;
          delta_plu_l_y = ((plu_dt_n_l_y - last_plu_dt_n_l_y) - (plu_dt_t_l_y - last_plu_dt_t_l_y)) * kDeltaAdjustInMM / 2;
          delta_plu_r = ((plu_dt_n_r - last_plu_dt_n_r) - (plu_dt_t_r - last_plu_dt_t_r)) * kDeltaAdjustInMM / 2;
          delta_plu_l = ((plu_dt_n_l - last_plu_dt_n_l) - (plu_dt_t_l - last_plu_dt_t_l)) * kDeltaAdjustInMM / 2;
          delta_plu_x = delta_plu_l_x - delta_plu_r_x;
          delta_plu_y = delta_plu_l_y - delta_plu_r_y;
          delta_plu = delta_plu_l - delta_plu_r;
        }

        if (delta_plu_r_x > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_r_x_count += 1;
        }
        if (delta_plu_l_x > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_l_x_count += 1;
        }
        if (delta_plu_r_y > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_r_y_count += 1;
        }
        if (delta_plu_l_y > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_l_y_count += 1;
        }
        if (delta_plu_r > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_r_count += 1;
        }
        if (delta_plu_l > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_l_count += 1;
        }
        if (delta_plu_x > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_x_count += 1;
        }
        if (delta_plu_y > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_y_count += 1;
        }
        if (delta_plu > kDeltaStrabismusThresholdInMM) {
          warn_delta_plu_count += 1;
        }
        
        // lastly update last_xxx values
        if (last_plu_dt_a_r < 0 || std::isinf(last_plu_dt_a_r)) {
          if (plu_dt_a_r > 0) {
            last_plu_dt_a_r_x = plu_dt_a_r_x;
            last_plu_dt_a_r_y = plu_dt_a_r_y;
            last_plu_dt_a_r = plu_dt_a_r;
            
            last_plu_dt_a_l_x = plu_dt_a_l_x;
            last_plu_dt_a_l_y = plu_dt_a_l_y;
            last_plu_dt_a_l = plu_dt_a_l;
            
            last_plu_dt_t_r_x = plu_dt_t_r_x;
            last_plu_dt_t_r_y = plu_dt_t_r_y;
            last_plu_dt_t_r = plu_dt_t_r;
            
            last_plu_dt_t_l_x = plu_dt_t_l_x;
            last_plu_dt_t_l_y = plu_dt_t_l_y;
            last_plu_dt_t_l = plu_dt_t_l;
            
            last_plu_dt_n_r_x = plu_dt_n_r_x;
            last_plu_dt_n_r_y = plu_dt_n_r_y;
            last_plu_dt_n_r = plu_dt_n_r;
            
            last_plu_dt_n_l_x = plu_dt_n_l_x;
            last_plu_dt_n_l_y = plu_dt_n_l_y;
            last_plu_dt_n_l = plu_dt_n_l;
          }
        }

        // left
        line = "";
        absl::StrAppend(&line, "left iris size : ", absl::StrFormat("% 5.1f", plu_left_iris_size), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "left ab : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_a_l, plu_dt_a_l_x, plu_dt_a_l_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "left tb : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_t_l, plu_dt_t_l_x, plu_dt_t_l_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "left nb : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_n_l, plu_dt_n_l_x, plu_dt_n_l_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "left delta : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", delta_plu_l, delta_plu_l_x, delta_plu_l_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "left count : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", warn_delta_plu_l_count, warn_delta_plu_l_x_count, warn_delta_plu_l_y_count));
        lines.emplace_back(line);

        // right
        line = "";
        absl::StrAppend(&line, "right iris size : ", absl::StrFormat("% 5.1f", plu_right_iris_size), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "right ab : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_a_r, plu_dt_a_r_x, plu_dt_a_r_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "right tb : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_t_r, plu_dt_t_r_x, plu_dt_t_r_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "right nb : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", plu_dt_n_r, plu_dt_n_r_x, plu_dt_n_r_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "right delta : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", delta_plu_r, delta_plu_r_x, delta_plu_r_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "right count : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", warn_delta_plu_r_count, warn_delta_plu_r_x_count, warn_delta_plu_r_y_count));
        lines.emplace_back(line);

        // total
        line = "";
        absl::StrAppend(&line, "iris  : ", absl::StrFormat("% 5.1f", show_plu_iris_size), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "delta : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", delta_plu, delta_plu_x, delta_plu_y), " mm");
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "count : ", absl::StrFormat("d% 5.1f, x% 5.1f, y% 5.1f", warn_delta_plu_count, warn_delta_plu_x_count, warn_delta_plu_y_count));
        lines.emplace_back(line);
        line = "";
        absl::StrAppend(&line, "const : ", absl::StrFormat("iris % 5.1f, calc % 5.1f, delta % 5.1f", kIrisSizeInMM, kDeltaAdjustInMM, kDeltaStrabismusThresholdInMM), " mm");
        lines.emplace_back(line);
      }
    }
  }

  AddTextRenderData(options, image_size, lines, render_data.get());

  cc->Outputs()
      .Tag(kRenderDataTag)
      .Add(render_data.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

void IrisToRenderDataCalculator::AddTextRenderData(
    const IrisToRenderDataCalculatorOptions& options,
    const std::pair<int, int>& image_size,
    const std::vector<std::string>& lines, RenderData* render_data) {
  int label_baseline_px = options.vertical_offset_px();
  float label_height_px =
      std::ceil(options.font_height_px() * kFontHeightScale);
  if (options.location() == IrisToRenderDataCalculatorOptions::TOP_LEFT) {
    label_baseline_px += label_height_px;
  } else if (options.location() ==
             IrisToRenderDataCalculatorOptions::BOTTOM_LEFT) {
    label_baseline_px += image_size.second - label_height_px * lines.size();
  }
  const auto label_left_px = options.horizontal_offset_px();
  for (int i = 0; i < lines.size(); ++i) {
    auto* label_annotation = render_data->add_render_annotations();
    label_annotation->set_thickness(2);

    label_annotation->mutable_color()->set_r(255);
    label_annotation->mutable_color()->set_g(0);
    label_annotation->mutable_color()->set_b(0);
    //
    auto* text = label_annotation->mutable_text();
    text->set_display_text(lines[i]);
    text->set_font_height(options.font_height_px());
    text->set_left(label_left_px);
    text->set_baseline(label_baseline_px + i * label_height_px);
    text->set_font_face(options.font_face());
  }
}

void IrisToRenderDataCalculator::RenderIris(
    const NormalizedLandmarkList& iris_landmarks,
    const IrisToRenderDataCalculatorOptions& options,
    const std::pair<int, int>& image_size, float iris_size,
    RenderData* render_data) {
  auto* oval_data_render = AddOvalRenderData(options, render_data);
  auto* oval_data = oval_data_render->mutable_oval();
  const float iris_radius = iris_size / 2.f;
  const auto& iris_center = iris_landmarks.landmark(0);

  oval_data->mutable_rectangle()->set_top(iris_center.y() -
                                          iris_radius / image_size.second);
  oval_data->mutable_rectangle()->set_bottom(iris_center.y() +
                                             iris_radius / image_size.second);
  oval_data->mutable_rectangle()->set_left(iris_center.x() -
                                           iris_radius / image_size.first);
  oval_data->mutable_rectangle()->set_right(iris_center.x() +
                                            iris_radius / image_size.first);
  oval_data->mutable_rectangle()->set_normalized(true);

  for (int i = 0; i < iris_landmarks.landmark_size(); ++i) {
    const NormalizedLandmark& landmark = iris_landmarks.landmark(i);
    auto* landmark_data_render = AddPointRenderData(options, render_data);
    auto* landmark_data = landmark_data_render->mutable_point();
    landmark_data->set_normalized(true);
    landmark_data->set_x(landmark.x());
    landmark_data->set_y(landmark.y());
  }
}

void IrisToRenderDataCalculator::GetLeftIris(const NormalizedLandmarkList& lds,
                                             NormalizedLandmarkList* iris) {
  // Center, top, bottom, left, right
  *iris->add_landmark() = lds.landmark(0);
  *iris->add_landmark() = lds.landmark(2);
  *iris->add_landmark() = lds.landmark(4);
  *iris->add_landmark() = lds.landmark(3);
  *iris->add_landmark() = lds.landmark(1);
}

void IrisToRenderDataCalculator::GetRightIris(const NormalizedLandmarkList& lds,
                                              NormalizedLandmarkList* iris) {
  // Center, top, bottom, left, right
  *iris->add_landmark() = lds.landmark(5);
  *iris->add_landmark() = lds.landmark(7);
  *iris->add_landmark() = lds.landmark(9);
  *iris->add_landmark() = lds.landmark(6);
  *iris->add_landmark() = lds.landmark(8);
}

RenderAnnotation* IrisToRenderDataCalculator::AddOvalRenderData(
    const IrisToRenderDataCalculatorOptions& options, RenderData* render_data) {
  auto* oval_data_annotation = render_data->add_render_annotations();
  oval_data_annotation->set_scene_tag(kOvalLabel);

  SetColor(oval_data_annotation, options.oval_color());
  oval_data_annotation->set_thickness(options.oval_thickness());
  return oval_data_annotation;
}

RenderAnnotation* IrisToRenderDataCalculator::AddPointRenderData(
    const IrisToRenderDataCalculatorOptions& options, RenderData* render_data) {
  auto* landmark_data_annotation = render_data->add_render_annotations();
  SetColor(landmark_data_annotation, options.landmark_color());
  landmark_data_annotation->set_thickness(options.landmark_thickness());

  return landmark_data_annotation;
}

}  // namespace mediapipe
