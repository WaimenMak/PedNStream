# Product Requirements Document: Crowd Control

## 1. Introduction

PyCrowd is a pedestrian traffic simulation tool based on the Link Transmission Model (LTM). It is designed to model and simulate the movement of pedestrians through a network of interconnected links and nodes, representing pathways and decision points. The simulation incorporates traffic dynamics to provide insights into pedestrian flow behavior under various conditions. And it can be wraped up as a training environment for Reinforcement Learning.

## 2. Goals

This project aims to achieve the following:
*   Accurately simulate pedestrian flow dynamics within a configurable network environment.
*   Model key elements of pedestrian movement, such as link capacities, travel speeds, densities, and queue formation.
*   Support various demand patterns for origin-destination (OD) flows to reflect real-world scenarios.
*   Enable pathfinding capabilities to determine pedestrian routes through the network.
*   Provide visualization tools to analyze simulation outputs, including network state, pedestrian densities, and flow rates over time.

## 3. Target Audience

*The simulation tool is built for pedestrian traffic practitioner.* 

## 4. Features

### 4.1. Implemented Features

The Crowd Control project currently includes the following implemented features:
*   **Core Simulation Engine (`src/LTM/network.py`):**
    *   Manages the overall simulation network, including initialization of network components (nodes and links).
    *   Executes the simulation on a step-by-step basis, updating network states over time.
*   **Link Module (`src/LTM/link.py`):**
    *   Models physical pathways (links) with attributes such as length, width, free-flow speed, capacity, and critical/jam densities.
    *   Simulates traffic dynamics on links, including calculating sending/receiving flows and updating pedestrian density and speed.
*   **Node Module (`src/LTM/node.py`):**
    *   Represents intersections or decision points within the network.
    *   Manages flow distribution between connected links based on defined logic and turning movements.
*   **Origin-Destination (OD) Management (`src/LTM/od_manager.py`):**
    *   Manages OD pairs and associated pedestrian demand.
    *   Includes a `DemandGenerator` capable of producing various demand patterns (e.g., constant flow, Gaussian peaks, sudden demand surges).
*   **Pathfinding Module (`src/LTM/path_finder.py`):**
    *   Determines available paths between origin and destination nodes within the network.
    *   Calculates turning fractions and probabilities at nodes to influence route choice.
*   **Visualization Utilities (`src/utils/visualizer.py`, `network_dashboard.py`):**
    *   Provides tools for visualizing network states at specific time steps.
    *   Offers capabilities to animate the simulation, showing the evolution of pedestrian movement (e.g., density, speed) over time.
    *   Includes an interactive dashboard (`network_dashboard.py`) for exploring simulation results.

### 4.2. Planned Features / Enhancements

Based on the current codebase, the following features or improvements are planned:

*   **LTM Module (`src/LTM/link.py`):**
    *   Integrate diffusion flow into the existing sending flow mechanism.
    *   Address and rectify issues in the flow release logic.
*   **Path Finding Module (`src/LTM/path_finder.py`):**
    *   Calibrate and fine-tune parameters for the utility function.
*   **Complete the Randomize features (`src/utils/env_loader.py`)**

### Planned Validation / Testing

* Validate the simulated sensor flow with real-world data to ensure accuracy.
* Validate the simulated sensor flow with a micro-simulation tool to ensure consistency.


## 5. Technical Considerations

*   Parameter calibration for pathfinding will be crucial for optimal performance.

## 6. Future Considerations / Open Questions

*   The specifics of the "diffusion flow" integration need to be detailed.
*   The exact nature of the issues with the "flow release logic" needs to be investigated and documented before fixing. 