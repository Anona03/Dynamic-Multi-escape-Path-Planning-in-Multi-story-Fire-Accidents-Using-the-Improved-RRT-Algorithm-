# Dynamic Multi-Escape Path Planning in Multi-Story Fire Environments

A Python implementation of an improved **RRT*M** algorithm for dynamic fire evacuation planning in multi-story buildings.

This project was developed as our undergraduate senior design project. It provides a grid-based fire simulation environment and a dynamic path planning algorithm capable of generating multiple independent evacuation routes while considering the temporal spread of fire and smoke.

---

## Overview

Traditional path planning algorithms often assume static environments, making them unsuitable for emergency evacuation where hazards continuously evolve.

This project proposes an **Improved RRT*M** algorithm that performs path planning in dynamic fire scenarios by integrating:

* Dynamic fire and smoke simulation
* Temporal obstacle avoidance
* Adaptive sampling
* Multi-path evacuation planning
* Constrained search regions

The system simulates multi-floor office buildings using grid maps and continuously updates evacuation paths as the environment changes.

---

## Features

* 🔥 Dynamic fire propagation simulation
* 🌫 Smoke diffusion modeling
* 🏢 Multi-floor building environment
* 🌳 Improved RRT*M algorithm
* ⏱ Temporal rewiring mechanism
* 📍 Adaptive confined search region
* 🚪 Multiple independent evacuation paths
* 📊 Visualization of planning process and evacuation results

---

## Project Structure

```text
.
├── Model_fire.py          # Fire environment (Floor A)
├── Model_fire2.py         # Fire environment (Floor B)
├── RRT-fast.py            # Improved RRT*M implementation
├── README.md
└── Design report.pdf      # Project report (optional)
```

---

## Algorithm Improvements

Compared with the original RRT*M algorithm, this implementation introduces several improvements.

### 1. Adaptive Sampling

Instead of uniformly sampling the entire environment, the algorithm dynamically restricts sampling inside an adaptive rectangular search region, reducing unnecessary exploration.

### 2. Temporal Rewiring

Unlike conventional rewiring based only on path length, the proposed method introduces **time** into the optimization process.

If a planned path becomes blocked due to fire spreading, the algorithm rewires the tree to generate a safer route with an earlier safe arrival time.

### 3. Multi-Path Planning

The planner supports

* Single source → multiple exits
* Multiple sources → single exit

while ensuring generated paths remain independent whenever possible.

### 4. Dynamic Fire Environment

The simulation continuously updates

* Fire spread
* Smoke diffusion
* Traversable regions

allowing the planner to react to changing hazards.

---

## Environment

The simulation represents buildings using a grid map.

Each grid cell can represent

* Free space
* Wall
* Fire
* Smoke

During simulation, fire and smoke propagate over time and dynamically modify the traversable space.

---

## Requirements

Python 3.10+

Main dependencies:

```bash
numpy
matplotlib
```

Install with

```bash
pip install numpy matplotlib
```

---

## Running the Project

Run the planner with

```bash
python RRT-fast.py
```

The program will

1. Load the building map.
2. Simulate fire and smoke spread.
3. Construct the improved RRT*M tree.
4. Generate evacuation paths.
5. Visualize the planning process.

---

## Example

The planner supports dynamic scenarios such as

* Fire spreading through corridors
* Smoke blocking previous routes
* Multiple evacuation exits
* Multi-floor environments

The generated path is automatically updated when the original route becomes unsafe.

---

## Research Background

This project was inspired by the original **RRT*M** algorithm and extends it for dynamic indoor fire evacuation.

The proposed framework combines

* Grid-based fire simulation
* Dynamic obstacle avoidance
* Temporal path optimization
* Multi-path evacuation planning

to improve evacuation efficiency in rapidly changing environments.

---

## Future Work

Potential extensions include

* 3D building models
* UAV-assisted rescue planning
* Human crowd simulation
* Reinforcement learning for adaptive planning
* GPU acceleration
* Real-time sensor integration

---

## Citation

If you use this project in your research, please cite:

> Dynamic Multi-Escape Path Planning in Multi-Story Fire Accidents Using the Improved RRT*M Algorithm, Undergraduate Senior Project, Sichuan University–Pittsburgh Institute, 2026.

---

## Authors

* Mouyong Jiang
* Jingwei Luo
* Junwei Fu

Sichuan University–Pittsburgh Institute

---

## License

This repository is released for academic and research purposes.

Please cite the project if you use any part of the code in your work.
