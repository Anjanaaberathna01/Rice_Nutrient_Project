# 🌱 Paddy-Care: AI-Driven Paddy Leaf Disease & Nutrient Deficiency Diagnostic System

![Flutter](https://img.shields.io/badge/Frontend-Flutter-blue?logo=flutter)
![Python](https://img.shields.io/badge/Backend-Python_Flask-green?logo=python)
![TensorFlow](https://img.shields.io/badge/AI-TensorFlow%20%7C%20Keras-orange?logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Active_Testing-brightgreen)

## 📌 Overview

Diagnosing paddy leaf diseases and nutrient deficiencies purely by visual inspection is highly challenging due to **Diagnostic Ambiguity** ("The Challenge of Diagnosing by Sight"). **Paddy-Care** is a robust, dual-platform (Mobile App & Web Dashboard) solution that overcomes this by integrating an advanced AI-driven image analysis pipeline with real-time environmental data.

Developed using a **Design Science Research (DSR)** methodology, this system not only analyzes leaf symptoms but also validates AI predictions using GN-division-level soil and weather data (Spatial Triangulation) to provide actionable, location-specific agricultural recommendations.

## ✨ Key Features & Architecture

### 1. 🧠 Hierarchical Cascading AI Strategya

The system utilizes a multi-stage CNN architecture to ensure high efficiency and accuracy (83% in open-field conditions):

- **Level 1: The Gatekeeper (MobileNetV2):** A lightweight model that instantly filters out non-paddy images, saving server resources and preventing false processing.
- **Level 2: The Main Ensemble:** Once validated by the gatekeeper, the image is passed to a powerful ensemble of **EfficientNetB0, DenseNet121, and InceptionV3** (integrated with SE Blocks for stressor segmentation). A **Soft-Voting** mechanism is used to calculate the final visual prediction.

### 2. 🌍 Contextual Validation (Spatial Triangulation)

To bridge the 17% gap in visual AI accuracy, the system uses GPS telemetry from the mobile app to fetch location-specific data (Divisional Secretariat/GN Division). Using **Pandas**, the system normalizes and cross-references local soil parameters (pH, N-P-K levels) and weather data with the AI's visual prediction to generate a highly validated final recommendation.

### 3. 📱 Dual-Platform Seamless Integration

- **Mobile App (Flutter):** For farmers in the field to capture images and receive instant, localized advice.
- **Web Dashboard (Flutter Web):** For agricultural officers and administrators to monitor regional disease heatmaps and trends.
- **Central Backend (Flask REST API):** Acts as the bridge, handling CORS, image preprocessing (NumPy), data normalization (Pandas), and serving the AI models (`0.0.0.0` configured for local hotspot testing).

## 🛠️ Technology Stack

- **Frontend:** Flutter (Dart), Image Picker, HTTP.
- **Backend:** Python, Flask, Flask-CORS.
- **AI & Machine Learning:** TensorFlow, Keras, OpenCV, NumPy (Image Normalization).
- **Data Processing:** Pandas (Environmental Data Normalization), Scikit-learn.

## 🚀 Getting Started (Local Development)

### Prerequisites

- Flutter SDK (3.x)
- Python (3.9+)
- Mobile Hotspot / Shared Wi-Fi network (for mobile-to-laptop API testing)

### Backend Setup (Flask API)

1. Clone the repository:
   ```bash
   git clone https://github.com/Anjanaaberathna01/Rice_Nutrient_Project.git
   cd paddy-care/backend
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```bash
   python app.py
   # Note: Runs on host='0.0.0.0', port=5000 for local network access.
   ```

### Frontend Setup (Flutter)

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   ```
2. Update the API URL:
   - Open `lib/config.dart` (or where your API URL is defined).
   - Change the localhost address to your machine's IPv4 address (e.g., `http://172.20.10.8:5000`).
3. Run the app on an emulator or physical device:
   ```bash
   flutter run
   ```
   _Note: Web platform image rendering is handled using `kIsWeb` to prevent `dart:io` assertion errors._

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
