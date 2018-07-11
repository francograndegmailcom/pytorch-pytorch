#ifndef CAFFE2_OPERATORS_UTILS_BOXES_H_
#define CAFFE2_OPERATORS_UTILS_BOXES_H_

#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

// Bounding box utils for generate_proposals_op
// Reference: detectron/lib/utils/boxes.py

namespace caffe2 {
namespace utils {

// Default value for minimum bounding box width and height after bounding box
//     transformation (bbox_transform()) in log-space
const float BBOX_XFORM_CLIP_DEFAULT = log(1000.0 / 16.0);
const float PI = 3.14159265358979323846;

// Forward transform that maps proposal boxes to ground-truth boxes using
//     bounding-box regression deltas.
// boxes: pixel coordinates of the bounding boxes
//     size (M, 4), format [x1; y1; x2; y2], x2 >= x1, y2 >= y1
// deltas: bounding box translations and scales
//     size (M, 4), format [dx; dy; dw; dh]
//     dx, dy: scale-invariant translation of the center of the bounding box
//     dw, dh: log-space scaling of the width and height of the bounding box
// weights: weights [wx, wy, ww, wh] for the deltas
// bbox_xform_clip: minimum bounding box width and height in log-space after
//     transofmration
// correct_transform_coords: Correct bounding box transform coordates. Set to
//     true to match the detectron code, set to false for backward compatibility
// return: pixel coordinates of the bounding boxes
//     size (M, 4), format [x1; y1; x2; y2]
// see "Rich feature hierarchies for accurate object detection and semantic
//     segmentation" Appendix C for more details
// reference: detectron/lib/utils/boxes.py bbox_transform()
template <class Derived1, class Derived2>
EArrXXt<typename Derived1::Scalar> bbox_transform_upright(
    const Eigen::ArrayBase<Derived1>& boxes,
    const Eigen::ArrayBase<Derived2>& deltas,
    const std::vector<typename Derived2::Scalar>& weights =
        std::vector<typename Derived2::Scalar>{1.0, 1.0, 1.0, 1.0},
    const float bbox_xform_clip = BBOX_XFORM_CLIP_DEFAULT,
    const bool correct_transform_coords = false) {
  using T = typename Derived1::Scalar;
  using EArrXX = EArrXXt<T>;
  using EArrX = EArrXt<T>;

  if (boxes.rows() == 0) {
    return EArrXX::Zero(T(0), deltas.cols());
  }

  CAFFE_ENFORCE_EQ(boxes.rows(), deltas.rows());
  CAFFE_ENFORCE_EQ(boxes.cols(), 4);
  CAFFE_ENFORCE_EQ(deltas.cols(), 4);

  EArrX widths = boxes.col(2) - boxes.col(0) + T(1.0);
  EArrX heights = boxes.col(3) - boxes.col(1) + T(1.0);
  auto ctr_x = boxes.col(0) + T(0.5) * widths;
  auto ctr_y = boxes.col(1) + T(0.5) * heights;

  auto dx = deltas.col(0).template cast<T>() / weights[0];
  auto dy = deltas.col(1).template cast<T>() / weights[1];
  auto dw =
      (deltas.col(2).template cast<T>() / weights[2]).cwiseMin(bbox_xform_clip);
  auto dh =
      (deltas.col(3).template cast<T>() / weights[3]).cwiseMin(bbox_xform_clip);

  EArrX pred_ctr_x = dx * widths + ctr_x;
  EArrX pred_ctr_y = dy * heights + ctr_y;
  EArrX pred_w = dw.exp() * widths;
  EArrX pred_h = dh.exp() * heights;

  T offset(correct_transform_coords ? 1.0 : 0.0);

  EArrXX pred_boxes = EArrXX::Zero(deltas.rows(), deltas.cols());
  // x1
  pred_boxes.col(0) = pred_ctr_x - T(0.5) * pred_w;
  // y1
  pred_boxes.col(1) = pred_ctr_y - T(0.5) * pred_h;
  // x2
  pred_boxes.col(2) = pred_ctr_x + T(0.5) * pred_w - offset;
  // y2
  pred_boxes.col(3) = pred_ctr_y + T(0.5) * pred_h - offset;

  return pred_boxes;
}

// Like bbox_transform_upright, but works on rotated boxes.
// boxes: pixel coordinates of the bounding boxes
//     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
// deltas: bounding box translations and scales
//     size (M, 5), format [dx; dy; dw; dh; da]
//     dx, dy: scale-invariant translation of the center of the bounding box
//     dw, dh: log-space scaling of the width and height of the bounding box
//     da: delta for angle in radians
// return: pixel coordinates of the bounding boxes
//     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
template <class Derived1, class Derived2>
EArrXXt<typename Derived1::Scalar> bbox_transform_rotated(
    const Eigen::ArrayBase<Derived1>& boxes,
    const Eigen::ArrayBase<Derived2>& deltas,
    const std::vector<typename Derived2::Scalar>& weights =
        std::vector<typename Derived2::Scalar>{1.0, 1.0, 1.0, 1.0},
    const float bbox_xform_clip = BBOX_XFORM_CLIP_DEFAULT) {
  using T = typename Derived1::Scalar;
  using EArrXX = EArrXXt<T>;
  using EArrX = EArrXt<T>;

  if (boxes.rows() == 0) {
    return EArrXX::Zero(T(0), deltas.cols());
  }

  CAFFE_ENFORCE_EQ(boxes.rows(), deltas.rows());
  CAFFE_ENFORCE_EQ(boxes.cols(), 5);
  CAFFE_ENFORCE_EQ(deltas.cols(), 5);

  const auto& ctr_x = boxes.col(0);
  const auto& ctr_y = boxes.col(1);
  const auto& widths = boxes.col(2);
  const auto& heights = boxes.col(3);
  const auto& angles = boxes.col(4);

  auto dx = deltas.col(0).template cast<T>() / weights[0];
  auto dy = deltas.col(1).template cast<T>() / weights[1];
  auto dw =
      (deltas.col(2).template cast<T>() / weights[2]).cwiseMin(bbox_xform_clip);
  auto dh =
      (deltas.col(3).template cast<T>() / weights[3]).cwiseMin(bbox_xform_clip);
  // Convert back to degrees
  auto da = deltas.col(4).template cast<T>() * 180.0 / PI;

  EArrXX pred_boxes = EArrXX::Zero(deltas.rows(), deltas.cols());
  // new ctr_x
  pred_boxes.col(0) = dx * widths + ctr_x;
  // new ctr_y
  pred_boxes.col(1) = dy * heights + ctr_y;
  // new width
  pred_boxes.col(2) = dw.exp() * widths;
  // new height
  pred_boxes.col(3) = dh.exp() * heights;
  // new angle
  pred_boxes.col(4) = da + angles;
  // TODO (viswanath): Normalize angle

  return pred_boxes;
}

template <class Derived1, class Derived2>
EArrXXt<typename Derived1::Scalar> bbox_transform(
    const Eigen::ArrayBase<Derived1>& boxes,
    const Eigen::ArrayBase<Derived2>& deltas,
    const std::vector<typename Derived2::Scalar>& weights =
        std::vector<typename Derived2::Scalar>{1.0, 1.0, 1.0, 1.0},
    const float bbox_xform_clip = BBOX_XFORM_CLIP_DEFAULT,
    const bool correct_transform_coords = false) {
  CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
  if (boxes.cols() == 4) {
    // Upright boxes
    return bbox_transform_upright(
        boxes, deltas, weights, bbox_xform_clip, correct_transform_coords);
  } else {
    // Rotated boxes with angle info
    return bbox_transform_rotated(boxes, deltas, weights, bbox_xform_clip);
  }
}

// Clip boxes to image boundaries
// boxes: pixel coordinates of bounding box, size (M * 4)
//
// For rotated boxes with angle support (M * 5), we don't clip and just
// return early. It's tricky to make the entire rectangular box fit within the
// image and still be able to not leave out pixels of interest.
// We rely on upstream ops like RoIAlignRotated safely handling such cases.
template <class Derived>
EArrXXt<typename Derived::Scalar>
clip_boxes(const Eigen::ArrayBase<Derived>& boxes, int height, int width) {
  CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
  if (boxes.cols() == 5) {
    // No clipping for rotated boxes.
    return boxes;
  }

  EArrXXt<typename Derived::Scalar> ret(boxes.rows(), boxes.cols());

  // x1 >= 0 && x1 < width
  ret.col(0) = boxes.col(0).cwiseMin(width - 1).cwiseMax(0);
  // y1 >= 0 && y1 < height
  ret.col(1) = boxes.col(1).cwiseMin(height - 1).cwiseMax(0);
  // x2 >= 0 && x2 < width
  ret.col(2) = boxes.col(2).cwiseMin(width - 1).cwiseMax(0);
  // y2 >= 0 && y2 < height
  ret.col(3) = boxes.col(3).cwiseMin(height - 1).cwiseMax(0);

  return ret;
}

// Only keep boxes with both sides >= min_size and center within the image.
// boxes: pixel coordinates of bounding box, size (M * 4)
// im_info: [height, width, img_scale]
// return: row indices for 'boxes'
template <class Derived>
std::vector<int> filter_boxes_upright(
    const Eigen::ArrayBase<Derived>& boxes,
    double min_size,
    const Eigen::Array3f& im_info) {
  CAFFE_ENFORCE_EQ(boxes.cols(), 4);

  // Scale min_size to match image scale
  min_size *= im_info[2];

  using T = typename Derived::Scalar;
  using EArrX = EArrXt<T>;

  EArrX ws = boxes.col(2) - boxes.col(0) + T(1);
  EArrX hs = boxes.col(3) - boxes.col(1) + T(1);
  EArrX x_ctr = boxes.col(0) + ws / T(2);
  EArrX y_ctr = boxes.col(1) + hs / T(2);

  EArrXb keep = (ws >= min_size) && (hs >= min_size) &&
      (x_ctr < T(im_info[1])) && (y_ctr < T(im_info[0]));

  return GetArrayIndices(keep);
}

// Similar to filter_boxes_upright but works for rotated boxes.
// boxes: pixel coordinates of the bounding boxes
//     size (M, 5), format [ctr_x; ctr_y; width; height; angle (in degrees)]
// im_info: [height, width, img_scale]
// return: row indices for 'boxes'
template <class Derived>
std::vector<int> filter_boxes_rotated(
    const Eigen::ArrayBase<Derived>& boxes,
    double min_size,
    const Eigen::Array3f& im_info) {
  CAFFE_ENFORCE_EQ(boxes.cols(), 5);

  // Scale min_size to match image scale
  min_size *= im_info[2];

  using T = typename Derived::Scalar;
  using EArrX = EArrXt<T>;

  const auto& x_ctr = boxes.col(0);
  const auto& y_ctr = boxes.col(1);
  const auto& ws = boxes.col(2);
  const auto& hs = boxes.col(3);

  EArrXb keep = (ws >= min_size) && (hs >= min_size) &&
      (x_ctr < T(im_info[1])) && (y_ctr < T(im_info[0]));

  return GetArrayIndices(keep);
}

template <class Derived>
std::vector<int> filter_boxes(
    const Eigen::ArrayBase<Derived>& boxes,
    double min_size,
    const Eigen::Array3f& im_info) {
  CAFFE_ENFORCE(boxes.cols() == 4 || boxes.cols() == 5);
  if (boxes.cols() == 4) {
    // Upright boxes
    return filter_boxes_upright(boxes, min_size, im_info);
  } else {
    // Rotated boxes with angle info
    return filter_boxes_rotated(boxes, min_size, im_info);
  }
}

} // namespace utils
} // namespace caffe2

#endif // CAFFE2_OPERATORS_UTILS_BOXES_H_
