# PartInfer: Enabling LLM Ineference On Edge Devices

## Overview

PartInfer, a neuron-level optimization framework that enables efficient LLM inference on edge devices by exploiting the task-specific activation patterns of neurons. 

![Overview](assets/overview.svg)

Our approach is split into 2 phase, offline (left) and online (right). During the offline phase, we run LLM Profiler to generate neurons files. During the offline phase, partial loading helps to reduce memory footprint while partial computation helps to reduce computational overhead.

## Demo

![Demo](assets/demo.mp4)

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

In order to run task evaluation using **PartInfer**, we use the [lm_eval](https://github.com/EleutherAI/lm-evaluation-harness) library.
We have built an automation script that runs evaluation of a certain model using a certain method on a number of datasets.

The default datasets used in our experiments are listed automatically in the file. You can add or remove datasets as you wish.
For a list of supported tasks, check the [lm-evaluation-harness tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks).

### Setting Up

1. **Choose your model**
   Set the variable `MODEL_NAME` to the name of the folder where your model is located.
   Make sure the model is placed in:

```
PartInfer/models/your_model
```

2. **Reproduce paper results**
Modify the following two variables: `METHOD` and `USE_PARTINFER_IMPROVEMENTS`.

- `METHOD="partinfer"`
  Runs the model using PartInfer.

- `METHOD="dense" & USE_PARTINFER_IMPROVEMENTS="True"`
  Runs PartInfer with partial loading only (no partial computation).

- `METHOD="dense" & USE_PARTINFER_IMPROVEMENTS="False"`
  Reference model (all neurons are loaded and fully computed).

- `METHOD="coreinfer" & USE_PARTINFER_IMPROVEMENTS="False"`
  Default CoreInfer approach.

- `METHOD="coreinfer" & USE_PARTINFER_IMPROVEMENTS="True"`
  CoreInfer's approach with our partial loading.

- `METHOD="coreinfer_random_loading"`
  CoreInfer with random partial loading.

Other variables are self-explanatory and are explained in more detail in the paper.


### Running the Script

After setting the variables, run the script:

```bash
bash evaluation_partinfer.sh
```

## Run Decoding Speed Evaluation

You can run the following command to see the decoding speed.

```bash
cd evaluation
python3 evaluate_speed.py --model_name $MODEL_NAME --checkpoint_path $PATH_TO_MODEL --num_tokens_to_generate $NUM_TOKENS --method $METHOD

# e.g. to run with PartInfer and Llama3.2-3B while loading 70% and computing 40% of neurons:
python3 evaluate_speed.py --model_name "llama3-3b" --checkpoint_path ../models/llama3-3b --num_tokens_to_generate 256 --method "partinfer" --loaded_neurons_percent 0.7 --sparsity -0.4
```

For more control on different percentages and neurons files, run the following command or refer to the paper:
```bash
python3 evaluate_speed.py --help
```
