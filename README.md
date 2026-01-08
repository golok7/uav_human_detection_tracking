# uav-human-detection-tracking
Project Description

This project presents an edge-optimized system for real-time human detection and tracking from unmanned aerial vehicle (UAV) imagery. The solution is designed specifically for deployment on resource-constrained embedded platforms such as Raspberry Pi, where computational efficiency, low memory usage, and real-time performance are critical.

The system employs a custom lightweight convolutional neural network (CNN) trained from scratch to detect human presence and estimate bounding boxes from aerial video frames. The model is tailored for aerial viewpoints and low-resolution inputs, avoiding reliance on large pretrained or open-source detection architectures. To maintain real-time operation, detection is performed intermittently and combined with a custom centroid-based tracking mechanism that ensures temporal consistency and stable identity assignment across frames.

The overall pipeline follows an edge-first design philosophy, prioritizing deployability, explainability, and robustness over brute-force accuracy. The project addresses aerial-specific challenges such as small object scale, camera motion, and dynamic backgrounds, making it suitable for applications including surveillance, search-and-rescue, and situational monitoring using UAV platforms.