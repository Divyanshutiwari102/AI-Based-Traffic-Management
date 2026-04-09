# AI-Based-Traffic-Management
An AI based traffic management system with real-time monitoring

## 🗒️ Overview

The Smart Adaptive Traffic Management System leverages AI and computer vision to optimize traffic flow at intersections. This system analyzes vehicle counts from video feeds, processes the data using machine learning models, and adjusts traffic signal timings to improve traffic flow.

## 📸 Screenshots

![1](screenshots/1.png)<br/><br/>
![2](screenshots/2.png)<br/><br/>
![3](screenshots/3.png)


## ✨ Features
- Vehicle Detection: YOLOv8-first detector pipeline (with safe fallback) and weighted multi-class density.
- Traffic Optimization: Webster-stabilized genetic optimizer using demand-ratio pressure (not inverted congestion).
- Async Processing API: `/upload` returns a `job_id`; `/status/<job_id>` exposes processing state and final result.
- Research-Ready Structure: Modular backend layout for detector, predictor, optimizer, metrics, and tasks.
- Web Interface: Upload 4 videos, track processing, and view optimized timings.

## 🧠 Architecture Diagram

```text
Cameras/Video Streams (N intersections)
        │
        ▼
[Frame Sampler + ROI + Calibration]
        │
        ▼
[Detector: YOLOv8] ──► [Tracker: DeepSORT/ByteTrack]
        │                         │
        └─► class counts          ├─► trajectories / queue / speed
                  │               ▼
                  └──────► [Feature Builder + Weighted Density]
                                   │
                                   ├─► [LSTM Predictor] -> short-horizon demand
                                   ▼
                         [State Constructor]
                                   │
                                   ▼
                         [PPO Policy Agent]
                                   │
                                   ▼
                     [GA/Webster Constraint Layer]
                                   │
                                   ▼
                    [Signal Plan Executor / SUMO]
                                   │
                                   ▼
                 [Metrics Service: AWT, queue, throughput, JFI]
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Nodejs
- OpenCV
- YOLOv8 model support (`ultralytics`) and PyTorch
- Required Python packages (listed in requirements.txt)

## 💻 Local Setup

Clone the repository:

```bash
git clone https://github.com/ashish0kumar/AI-Based-Traffic-Management.git
cd AI-Based-Traffic-Management
```

Start the backend server:

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Start the frontend server:
```bash
cd frontend
npm install
npm start
```

Upload Traffic Videos: <br/>
Use the web interface to upload 4 traffic videos. The backend returns a `job_id`, and the frontend polls status until optimized timings are ready.

## 🙏 Acknowledgments

- YOLOv8 / Ultralytics: For vehicle detection.
- OpenCV: For video processing.
- Genetic Algorithm + Webster constraints: For optimizing safe traffic light timings.
