# F1 Debris Detection on Track using YOLOv11n

This project focuses on detecting debris and other critical objects on Formula 1 (F1) racing tracks using real-time computer vision. By leveraging YOLOv11n (You Only Look Once version 11n) for object detection, the model identifies various objects such as debris, recovery vehicles, and marshals on the track. The primary goal is to improve race safety by automatically detecting potential hazards during live races.

## Project Overview

The primary objective of this project is to create a robust and accurate object detection system capable of identifying debris on the track, which poses significant risks to drivers during a race. The system is designed using the YOLOv11n architecture and trained on a labeled dataset provided by Roboflow. Key components of the project include:

- **Roboflow's Pre-annotated Dataset**: A collection of images annotated with various objects, including debris, recovery vehicles, and marshals. This dataset is used to train the object detection model.
- **YOLOv11n Architecture**: YOLOv11n is an advanced version of the YOLO model, optimized for real-time object detection with high accuracy and efficiency. The model is well-suited for applications like detecting debris on the racing track.
- **Training and Fine-Tuning**: The dataset is used to train the YOLO model. The training process adjusts the model’s weights and parameters to improve its ability to identify objects like debris, marshals, and recovery vehicles on the track.

## Core Principles

### 1. **Object Detection with YOLOv11n**
   YOLOv11n is a state-of-the-art object detection model that excels at simultaneously detecting and localizing multiple objects within an image. Unlike traditional models that process images sequentially, YOLO performs all tasks in a single pass, making it exceptionally fast and accurate for real-time applications.

   In this project, YOLOv11n is trained to detect three primary objects:
   - **Debris**: Obstacles that could pose a risk to the race, such as tire fragments or broken parts.
   - **Marshals**: Track officials who manage the race environment and ensure safety.
   - **Recovery Vehicles**: Vehicles that are deployed to assist with removing debris or providing aid during an emergency.

### 2. **Dataset from Roboflow**
   The model is trained using a dataset sourced from Roboflow, which provides high-quality, labeled images containing annotated instances of debris, marshals, and recovery vehicles. The dataset is formatted specifically for training object detection models, ensuring effective learning of the relevant features needed for identifying these objects in images from real-world F1 races.

### 3. **Training the YOLOv11n Model**
   The training process involves using the labeled dataset to train the YOLOv11n model, which is designed to predict both the bounding boxes (for localization) and class labels (for identification) of objects. The training process consists of the following steps:
   - **Forward Pass**: Images are passed through the network to generate predictions (bounding boxes and class labels).
   - **Loss Computation**: A loss function evaluates the difference between the model’s predictions and the true labels from the dataset.
   - **Backpropagation**: The loss is used to update the model’s weights using an optimization algorithm like gradient descent, improving the model’s ability to detect objects over time.

   The model is trained for 50 epochs, with images resized to 1080px, using the YOLOv11n architecture, which is optimized for this type of task.

### 4. **Model Evaluation**
   After training, the model is evaluated using a separate validation dataset. The evaluation involves comparing the predicted bounding boxes and class labels to the ground truth labels to assess the model’s performance. The key evaluation metrics include:
   - **Precision**: The ratio of correctly predicted positive instances to the total predicted positives.
   - **Recall**: The ratio of correctly predicted positive instances to all actual positive instances.
   - **mAP (mean Average Precision)**: An overall performance metric that averages precision at different recall levels, providing a comprehensive measure of the model’s detection ability.

   The results from the confusion matrix, recall-confidence curves, and precision-confidence curves further validate the model’s performance in detecting debris, marshals, and recovery vehicles.

### 5. **Real-Time Inference**
   Once trained, the YOLOv11n model can make predictions on unseen images in real-time. The model detects debris, marshals, and recovery vehicles by drawing bounding boxes and providing confidence scores for each detected object. Given YOLOv11n’s real-time performance, the model can be deployed for on-the-fly analysis during F1 races to assist in safety monitoring.

   **Performance on 1080p F1 POV Video:**  
   The trained model was tested on a real-world 1080p F1 point-of-view (POV) video. The model successfully detected high-speed flying debris from the tires as well as stationary debris on the track. The ability to track high-speed flying debris, such as tire fragments, demonstrates the model’s robustness in high-speed, dynamic environments, crucial for race safety. This testing proves the model's capability to detect hazards in real-time, even in the challenging conditions of an F1 race.

## Results and Performance

The performance of the model is evaluated based on multiple metrics, including precision, recall, and mAP. These metrics give us insights into how well the model detects the objects critical for safety during F1 races. Below is an analysis based on the curves and confusion matrices:
- **Debris**: The model achieved reasonable recall and precision for detecting debris, with some room for improvement in recall.
- **Marshal**: The detection of marshals was relatively accurate, with high precision and recall values, indicating the model’s ability to recognize marshals under various conditions.
- **Recovery Vehicle**: The detection of recovery vehicles showed the highest performance in terms of both precision and recall, suggesting that the model is very reliable in identifying these critical objects.

### Confusion Matrix Analysis
   The confusion matrix provides a detailed breakdown of the model's ability to differentiate between different object classes. Below is the normalized confusion matrix for the evaluation:

   The model shows strong performance in detecting recovery vehicles and marshals, with some misclassifications between debris and other objects (e.g., marshals and background).

### Precision-Recall and Confidence Curves
   - **Recall-Confidence Curve**: Shows the recall for different confidence thresholds. The recall for recovery vehicles is particularly strong, with the model consistently performing well across confidence thresholds.
   - **Precision-Confidence Curve**: Shows how precision changes as the confidence threshold varies. The precision for all object classes increases significantly at higher confidence levels, with recovery vehicles achieving near-perfect precision.

## Conclusion

This project demonstrates the capabilities of YOLOv11n for detecting critical objects like debris, marshals, and recovery vehicles on F1 tracks in real-time. By employing this advanced object detection model, the system can greatly enhance safety during races by swiftly identifying potential hazards.

Testing the model on a 1080p F1 POV video showed its ability to detect high-speed flying debris from the tires, as well as debris on the track, highlighting its effectiveness in real-world racing conditions.

Future work could involve expanding the dataset to include more diverse environmental conditions, enhancing the video quality, refining the model, and using more advanced models with high-definition capabilities to reduce misclassifications between similar objects like debris and marshals. Moreover, further fine-tuning of the model could improve its performance in challenging lighting or weather conditions on the track.
