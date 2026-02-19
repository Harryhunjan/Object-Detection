# Object Detection Project

## 📌 Overview

This project implements an **Object Detection System** using **Python, OpenCV, and NumPy**. It is capable of detecting specified object categories in images, videos, or real-time webcam input. The system applies bounding box techniques to highlight detected objects and provides their spatial positions.

## 🚀 Features

* Detects objects in **images, videos, and live webcam feed**
* Uses **OpenCV** for image processing and visualization
* Employs **bounding boxes** to locate and track detected objects
* Efficient real-time detection using **NumPy optimization**
* Modular and extensible for integrating with other ML/DL models

## 🛠️ Technologies Used

* **Python 3.x**
* **OpenCV** (cv2)
* **NumPy**

## 📂 Project Structure

```
Project/
│── main.py          # Main script to run object detection
│── utils.py         # Utility functions for processing
│── requirements.txt # Dependencies
│── samples/         # Sample images/videos for testing
│── README.md        # Project documentation
```

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Harryhunjan/Object-Detection.git
   cd Object-Detection/Project
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the detection script:

   ```bash
   python main.py
   ```

## 🎯 Usage

* To run detection on an image:

  ```bash
  python main.py --image samples/test.jpg
  ```
* To run detection on a video:

  ```bash
  python main.py --video samples/test.mp4
  ```
* To run real-time detection using a webcam:

  ```bash
  python main.py --webcam
  ```

## 📊 Output

* Displays input with bounding boxes around detected objects
* Shows coordinates (x, y, width, height) of each detected object

## 📝 Future Enhancements

* Integration with **deep learning models (YOLO, SSD, Faster R-CNN)**
* Multi-object tracking
* Performance improvements for large-scale datasets
* GUI-based user interface

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License.
