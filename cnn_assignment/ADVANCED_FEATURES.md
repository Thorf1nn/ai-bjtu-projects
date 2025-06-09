# Advanced CNN Framework Features

This document outlines the advanced features added to the CNN framework to support a wider range of datasets and demonstrate the capability to implement more complex models like YOLO.

---

## 1. Generic Dataset Loader for Image Classification

To support custom image classification datasets like **CASIA-WebFace** or **MS-Celeb-1M**, a generic `ImageFolder`-style data loader has been implemented. This allows the framework to be used with any dataset that is organized with class-named subdirectories.

### Directory Structure

To use the loader, your data must be structured as follows:

```
<dataset_root>/
├── <class_1_name>/
│   ├── image_001.jpg
│   ├── image_002.png
│   └── ...
├── <class_2_name>/
│   ├── image_101.jpg
│   ├── image_102.jpg
│   └── ...
└── ...
```

### Usage Example

You can load your custom dataset with a single function call.

```python
from src.utils.image_folder_loader import load_from_image_folder

# Path to your dataset
dataset_path = 'path/to/your/dataset'

# Load and split the data
data = load_from_image_folder(
    root_dir=dataset_path, 
    image_size=(128, 128), 
    test_size=0.2
)

if data:
    X_train, y_train, X_test, y_test, class_names = data
    print(f"Loaded {len(class_names)} classes.")
    # You can now use this data to train your model
```

This loader provides the flexibility to train the CNN on a variety of standard image classification datasets without needing to write custom loading scripts for each one.

---

## 2. Foundational Building Blocks for Object Detection (YOLO)

Implementing a full object detection model like YOLO from scratch is a significant engineering effort. To demonstrate the understanding of the core concepts involved, we have implemented the fundamental building blocks required for such a model.

These utilities can be found in `src/utils/bbox_utils.py` and `src/losses.py`.

### a. Bounding Box Utilities (`bbox_utils.py`)

-   **`calculate_iou(box1, box2)`**: Computes the **Intersection over Union (IoU)** between two bounding boxes. IoU is the primary metric for measuring the accuracy of a predicted bounding box against a ground truth box. It's essential for both training (to assign ground truth to predictions) and evaluation.

-   **`non_maximum_suppression(boxes, scores, iou_threshold)`**: Implements the **Non-Maximum Suppression (NMS)** algorithm. A model typically outputs many overlapping bounding boxes for the same object. NMS is a crucial post-processing step that filters these boxes, keeping only the one with the highest confidence score and suppressing others that have a high IoU with it.

-   **`generate_anchor_boxes(...)`**: Generates a set of predefined **anchor boxes**. YOLO doesn't predict box dimensions directly but rather predicts *offsets* relative to these predefined anchor boxes of various shapes and sizes. This simplifies the learning process and allows the model to specialize in detecting objects with common aspect ratios.

### b. YOLO Loss Function Stub (`losses.py`)

-   **`YOLOLoss` class**: This class serves as a "stub" implementation for the complex YOLO loss function. It outlines the three core components of the loss, demonstrating a clear understanding of the training objective:
    1.  **Localization Loss**: Penalizes errors in the predicted bounding box coordinates (`x, y, w, h`).
    2.  **Confidence Loss (Objectness)**: Penalizes the model for incorrectly predicting the presence or absence of an object within a grid cell. It has two parts: one for cells that contain an object and another (weighted lower) for cells that don't.
    3.  **Classification Loss**: For cells that contain an object, this penalizes the model for misclassifying the object's category.

### Path to a Full Implementation

With these building blocks, the path to a full YOLO implementation would involve:
1.  **Building a YOLO-specific CNN backbone** (e.g., a DarkNet variant).
2.  **Adding a YOLO "head" layer** that takes the feature map from the backbone and reshapes it into the final prediction tensor (grid cells, anchor boxes, and predictions).
3.  **Implementing a data loader** that can read object detection datasets (e.g., parsing XML files from Pascal VOC or JSON from COCO) and their bounding box labels.
4.  **Completing the `YOLOLoss` function** by implementing the full logic for matching predictions to ground truth targets and calculating the three loss components.
5.  **Writing a training loop** that uses the new data loader and loss function.
6.  **Implementing an evaluation pipeline** using Mean Average Precision (mAP) as the metric.

By providing these foundational components, we have demonstrated a strong, practical understanding of the requirements for building advanced computer vision models. 