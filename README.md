# PartInfer: Enabling LLM Ineference On Edge Devices

## Overview

PartInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices by exploiting the task-specific activation patterns of neurons. 

![Overview](assets/overview.svg)

Our approach is split into 2 phase, offline (left) and online (right). During the offline phase, we run LLM Profiler to generate neurons files. During the offline phase, partial loading helps to reduce memory footprint while partial computation helps to reduce computational overhead.

## Demo

![Overview](assets/demp.mp4)

## Install

This project uses a Python virtual environment to manage dependencies. Follow the steps below to reproduce the same environment.

### 1. Clone the Repository

```bash
git clone https://github.com/Huzifa1/PartInfer
cd PartInfer
```

### 2. Create a Virtual Environment

Make sure you have Python (>=3.x) installed.

```bash
python3 -m venv .partinfer-venv
```

### 3. Activate the Virtual Environment

**Linux / macOS**:
    
```bash
source venv/bin/activate
```
    
**Windows (PowerShell)**:
    
```powershell
.\venv\Scripts\Activate
```
    

### 4. Install Dependencies

The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Notes
    
To deactivate the virtual environment:
    
```bash
deactivate
```

## Create Neurons Files

## Run Inference

## Run Task Evaluation

## Run Decoding Speed Evaluation