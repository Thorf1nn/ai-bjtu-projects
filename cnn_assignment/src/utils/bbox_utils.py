import numpy as np

def calculate_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Each box is expected to be in the format [x1, y1, x2, y2].

    Args:
        box1 (list or np.ndarray): The first bounding box.
        box2 (list or np.ndarray): The second bounding box.

    Returns:
        float: The IoU value, between 0 and 1.
    """
    # Get the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the union
    union_area = box1_area + box2_area - inter_area

    # Compute the IoU
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou


def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Applies Non-Maximum Suppression (NMS) to filter out overlapping bounding boxes.

    Args:
        boxes (np.ndarray): An array of bounding boxes, shape (N, 4).
        scores (np.ndarray): An array of confidence scores for each box, shape (N,).
        iou_threshold (float): The threshold for IoU to consider boxes as overlapping.

    Returns:
        list: A list of indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    # Sort boxes by their confidence scores in descending order
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # The box with the highest score is always kept
        i = order[0]
        keep.append(i)

        # Get the IoU of the current box with all other boxes
        ious = np.array([calculate_iou(boxes[i], boxes[j]) for j in order[1:]])

        # Find indices of boxes that have an IoU less than the threshold
        inds_to_keep = np.where(ious <= iou_threshold)[0]

        # Update the order to only include the boxes to keep
        order = order[inds_to_keep + 1]

    return keep


def generate_anchor_boxes(input_width, input_height, feature_map_width, feature_map_height, anchor_priors):
    """
    Generates anchor boxes for a given feature map size.

    Args:
        input_width (int): Width of the original model input image.
        input_height (int): Height of the original model input image.
        feature_map_width (int): Width of the feature map from the CNN backbone.
        feature_map_height (int): Height of the feature map.
        anchor_priors (np.ndarray): An array of shape (num_anchors, 2) containing
                                    the prior widths and heights of anchor boxes.

    Returns:
        np.ndarray: An array of anchor boxes in [x_center, y_center, width, height]
                    format, shape (feature_map_width, feature_map_height, num_anchors, 4).
    """
    num_anchors = len(anchor_priors)
    stride_w = input_width / feature_map_width
    stride_h = input_height / feature_map_height
    
    # Create a grid of center points for each cell in the feature map
    grid_x = np.arange(feature_map_width) * stride_w + stride_w / 2
    grid_y = np.arange(feature_map_height) * stride_h + stride_h / 2
    center_x, center_y = np.meshgrid(grid_x, grid_y)
    
    # Expand dims to broadcast with anchors
    center_x = center_x[:, :, np.newaxis]
    center_y = center_y[:, :, np.newaxis]
    
    # Create anchor boxes for each grid cell
    anchors = np.zeros((feature_map_height, feature_map_width, num_anchors, 4))
    anchors[:, :, :, 0] = center_x
    anchors[:, :, :, 1] = center_y
    anchors[:, :, :, 2] = anchor_priors[:, 0]
    anchors[:, :, :, 3] = anchor_priors[:, 1]
    
    return anchors 