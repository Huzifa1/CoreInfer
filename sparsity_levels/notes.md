
# NEW SPARSITY_LEVELS METHOD

## Llama3-3b:

- For the comparison with CoreInfer, sparsity = 0.4 and token_sparsity = 0.2 has been used
- CoreInfer actual sparsity for all of them was 0.4
- When determining these sparsity levels, the goal sparsity was set to 0.4 for all datasets
- These params are evaluated on 100 samples from the corresponding dataset.

### TruthfulQA:

|  Params                | Actual Sparsity    | BLEU Max      |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.399              | 4.12          |
| [0.2, 0.8, 0.48, 100]  | 0.408              | 3.85          |            
| [0.25, 0.75, 0.47, 100]| 0.395              | 3.73          |
| [0.3, 0.7, 0.45, 100]  | 0.404              | 4.20          |
| [0.4, 0.6, 0.42, 100]  | 0.396              | 3.26          |
| [0.45, 0.55, 0.4, 100] | 0.403              | 4.72          |
| [0.55, 0.45, 0.37, 100]| 0.395              | 3.62          |
| [0.6, 0.4, 0.35, 100]  | 0.403              | 2.78          |

- CoreInfer got **6.33**
- Then, I have picked the 6th configuration (4.72) and run it on the whole dataset and got a score of **6.627** (104.6%)
- Then I used the AVG, which has actual sparsity = 0.4 and run it on the whole dataset and got a score of **6.446** (101.8%)


### BertaQA_en:

|  Params                | Actual Sparsity    | Accuracy      |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.394              | 0.52          |
| [0.2, 0.8, 0.48, 100]  | 0.407              | 0.52          |            
| [0.25, 0.75, 0.47, 100]| 0.393              | 0.52          |
| [0.3, 0.7, 0.45, 100]  | 0.402              | 0.52          |
| [0.4, 0.6, 0.42, 100]  | 0.397              | 0.52          |
| [0.45, 0.55, 0.4, 100] | 0.408              | 0.52          |
| [0.5, 0.5, 0.39, 100]  | 0.393              | 0.52          |
| [0.55, 0.45, 0.37, 100]| 0.402              | 0.52          |

- CoreInfer got **0.575**
- Then, I have picked the 4th configuration (sparsity = 0.402) and run it on the whole dataset and got a score of **0.575** (100%)
- Then I used the AVG, which has actual sparsity = 0.4 and run it on the whole dataset and got a score of **0.575** (100%)


### SquadV2:

|  Params                | Actual Sparsity    | Exact Match   |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.397              | 14.0          |
| [0.25, 0.75, 0.47, 100]| 0.401              | 16.0          |            
| [0.35, 0.65, 0.44, 100]| 0.404              | 18.0          |
| [0.4, 0.6, 0.43, 100]  | 0.391              | 20.0          |
| [0.45, 0.55, 0.41, 100]| 0.406              | 18.0          |
| [0.5, 0.5, 0.4, 100]   | 0.393              | 19.0          |
| [0.55, 0.45, 0.38, 100]| 0.407              | 19.0          |
| [0.6, 0.4, 0.37, 100]  | 0.396              | 16.0          |

- CoreInfer got **16.9**
- Then, I have picked the 4th configuration (20.0) and run it on the whole dataset and got a score of **15.7** (92.89%)
- Then I used the AVG, which has actual sparsity = 0.4 and run it on the whole dataset and got a score of **16.4** (97.04%)


### CommonSenseQA:

|  Params                | Actual Sparsity    | Accuracy      |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.398              | 0.59          |
| [0.25, 0.75, 0.47, 100]| 0.395              | 0.59          |            
| [0.3, 0.7, 0.45, 100]  | 0.406              | 0.59          |
| [0.35, 0.65, 0.44, 100]| 0.390              | 0.59          |
| [0.4, 0.6, 0.42, 100]  | 0.399              | 0.59          |
| [0.45, 0.55, 0.4, 100] | 0.409              | 0.59          |
| [0.5, 0.5, 0.39, 100]  | 0.393              | 0.59          |
| [0.55, 0.45, 0.37, 100]| 0.403              | 0.59          |

- CoreInfer got **0.653**
- Then, I have picked the 5th configuration (sparsity = 0.399) and run it on the whole dataset and got a score of **0.653** (100%)
- Then I used the AVG, which has actual sparsity = 0.4 and run it on the whole dataset and got a score of **0.653** (100%)


### TriviaQA:

|  Params                | Actual Sparsity    | Exact Match   |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.398              | 0.19          |
| [0.2, 0.8, 0.48, 100]  | 0.409              | 0.19          |            
| [0.25, 0.75, 0.47, 100]| 0.395              | 0.20          |
| [0.3, 0.7, 0.45, 100]  | 0.405              | 0.20          |
| [0.35, 0.65, 0.44, 100]| 0.390              | 0.18          |
| [0.4, 0.6, 0.42, 100]  | 0.399              | 0.18          |
| [0.5, 0.5, 0.39, 100]  | 0.395              | 0.18          |
| [0.55, 0.45, 0.37, 100]| 0.403              | 0.17          |

- CoreInfer got **0.262**
- Then, I have picked the 3rd configuration (sparsity = 0.409) and run it on the whole dataset and got a score of **0.264** (100.7%)
- Then I used the AVG, which has actual sparsity = 0.4 and run it on the whole dataset and got a score of **0.257** (98.09%)


### WMT16-DE-EN:

|  Params                | Actual Sparsity    | BLEU          |  
|------------------------|:------------------:|:-------------:|
| [0.15, 0.85, 0.5, 100] | 0.393              | 6.11          |
| [0.2, 0.8, 0.48, 100]  | 0.404              | 6.87          |            
| [0.25, 0.75, 0.47, 100]| 0.391              | 4.79          |
| [0.3, 0.7, 0.45, 100]  | 0.402              | 7.00          |
| [0.4, 0.6, 0.42, 100]  | 0.398              | 4.48          |
| [0.5, 0.5, 0.39, 100]  | 0.393              | 3.24          |
| [0.55, 0.45, 0.37, 100]| 0.405              | 4.04          |
| [0.6, 0.4, 0.36, 100]  | 0.390              | 3.88          |

- CoreInfer got **4.185**
- Then, I have picked the 4th configuration (7.00) and run it on the whole dataset and got a score of **3.81** (91.03%)
- Then I used the AVG, which has actual sparsity = 0.398 and run it on the whole dataset and got a score of **3.485** (83.27%)


## Llama3b with 10% less sparsity


- CoreInfer actual sparsity for all of them was 0.4
- When determining these sparsity levels, the goal sparsity was set to 0.36 (-10%) for all datasets
- These params are evaluated on 100 samples from the corresponding dataset.


### wmt16-de-en

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.354 | 3.15 |
| 1 | 0.365 | 3.85 |
| 2 | 0.351 | 3.25 |
| 3 | 0.36 | 2.67 |
| 4 | 0.358 | 2.96 |
| 5 | 0.355 | 3.08 |
| 6 | 0.365 | 3.25 |
| 7 | 0.351 | 2.62 |

- CoreInfer got **2.58**
- I used the AVG, which has actual sparsity = 0.357 and run it on the whole dataset and got a score of **2.293** (88.87%)


### squadv2

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.36 | 16.0 |
| 1 | 0.364 | 15.0 |
| 2 | 0.353 | 17.0 |
| 3 | 0.366 | 19.0 |
| 4 | 0.356 | 15.0 |
| 5 | 0.369 | 18.0 |
| 6 | 0.359 | 16.0 |
| 7 | 0.364 | 14.0 |
| 8 | 0.353 | 14.0 |

- CoreInfer got **16.5**
- I used the AVG, which has actual sparsity = 0.36 and run it on the whole dataset and got a score of **13.7** (83.03%)


### truthfulqa_gen

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.365 | 4.09 |
| 1 | 0.358 | 5.07 |
| 2 | 0.365 | 4.82 |
| 3 | 0.35 | 3.46 |
| 4 | 0.357 | 4.13 |
| 5 | 0.364 | 2.52 |
| 6 | 0.35 | 4.5 |
| 7 | 0.357 | 3.45 |
| 8 | 0.363 | 3.49 |

- CoreInfer got **5.193**
- I used the AVG, which has actual sparsity = 0.359 and run it on the whole dataset and got a score of **6.158** (118.58%)



### bertaqa_en

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.358 | 0.52 |
| 1 | 0.367 | 0.52 |
| 2 | 0.352 | 0.52 |
| 3 | 0.364 | 0.52 |
| 4 | 0.358 | 0.52 |
| 5 | 0.368 | 0.52 |
| 6 | 0.355 | 0.52 |
| 7 | 0.363 | 0.52 |
| 8 | 0.351 | 0.52 |

- CoreInfer got **0.575**
- I used the AVG, which has actual sparsity = 0.36 and run it on the whole dataset and got a score of **0.575** (100%)



### commonsense_qa

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.363 | 0.59 |
| 1 | 0.358 | 0.59 |
| 2 | 0.366 | 0.59 |
| 3 | 0.352 | 0.59 |
| 4 | 0.361 | 0.59 |
| 5 | 0.356 | 0.59 |
| 6 | 0.365 | 0.59 |
| 7 | 0.351 | 0.59 |

- CoreInfer got **0.653**
- I used the AVG, which has actual sparsity = 0.359 and run it on the whole dataset and got a score of **0.653** (100%)



### triviaqa

| Levels | Actual Sparsity | Metric Value |
| --- | --- | --- |
| 0 | 0.363 | 0.2 |
| 1 | 0.35 | 0.18 |
| 2 | 0.359 | 0.18 |
| 3 | 0.368 | 0.17 |
| 4 | 0.355 | 0.19 |
| 5 | 0.363 | 0.19 |
| 6 | 0.356 | 0.19 |
| 7 | 0.365 | 0.18 |
| 8 | 0.352 | 0.19 |

- CoreInfer got **0.248**
- I used the AVG, which has actual sparsity = 0.359 and run it on the whole dataset and got a score of **0.246** (99.19%)


## OPT-6.7b:

- For the comparison with CoreInfer, sparsity = 0.4 and token_sparsity = 0.2 has been used

### TruthfulQA:

- CoreInfer actual sparsity for all of them was 0.26
- When determining these sparsity levels, the goal sparsity was set to 0.26:

|  Params                | Actual Sparsity    | BLEU Max      |  
|------------------------|:------------------:|:-------------:|
| [0.1, 0.9, 0.035, 100] | 0.254              | 24.88         |
| [0.2, 0.8, 0.035, 100] | 0.249              | 24.72         |            
| [0.3, 0.7, 0.03, 100]  | 0.261              | 24.55         |
| [0.4, 0.6, 0.03, 100]  | 0.255              | 24.64         |
| [0.5, 0.5, 0.025, 100] | 0.262              | 24.51         |
| [0.5, 0.5, 0.03, 100]  | 0.249              | 25.35         |
| [0.6, 0.4, 0.025, 100] | 0.256              | 24.87         |
| [0.7, 0.3, 0.02, 100]  | 0.262              | 24.60         |
| [0.7, 0.3, 0.025, 100] | 0.250              | 25.43         |
| [0.8, 0.2, 0.02, 100]  | 0.256              | 25.17         |
| [0.9, 0.1, 0.015, 100] | 0.261              | 24.84         |
| [0.9, 0.1, 0.02, 100]  | 0.250              | 26.13         |
| [1.0, 0.0, 0.015, 100] | 0.256              | 25.21         |

- CoreInfer got **25.19**
- Then, I have picked the 12th configuration (26.13) and run it on the whole dataset and got a score of **25.64** (101.7%)
- Then I used the AVG, which has actual sparsity = 0.256 and run it on the whole dataset and got a score of **25.22** (100.1%)


### BertaQA_en:

- CoreInfer actual sparsity for all of them was 0.32
- When determining these sparsity levels, the goal sparsity was set to 0.32:

|  Params                | Actual Sparsity    | Accuracy      |  
|------------------------|:------------------:|:-------------:|
| [0.4, 0.6, 0.02, 100]  | 0.315              | 0.36          |
| [0.5, 0.5, 0.02, 100]  | 0.309              | 0.36          |            
| [0.7, 0.3, 0.015, 100] | 0.318              | 0.36          |
| [0.8, 0.2, 0.015, 100] | 0.314              | 0.36          |
| [0.9, 0.1, 0.015, 100] | 0.309              | 0.36          |
| [1.0, 0.0, 0.01, 100]  | 0.318              | 0.36          |

- CoreInfer got **0.326**
- Then, I have picked the 3rd configuration (sparsity = 0.318) and run it on the whole dataset and got a score of **0.322** (%98.7)
- Then I used the AVG, which has actual sparsity = 0.314 and run it on the whole dataset and got a score of **0.322** (%98.7)


### SquadV2:

- CoreInfer actual sparsity for all of them was 0.4
- When determining these sparsity levels, the goal sparsity was set to 0.4:

|  Params                | Actual Sparsity    | Exact Match   |  
|------------------------|:------------------:|:-------------:|
| [0.2, 0.8, 0.015, 100] | 0.390              | 28.0          |
| [0.3, 0.7, 0.015, 100] | 0.397              | 27.0          |            
| [0.4, 0.6, 0.015, 100] | 0.402              | 27.0          |
| [0.5, 0.5, 0.015, 100] | 0.403              | 27.0          |
| [0.6, 0.4, 0.015, 100] | 0.406              | 28.0          |
| [0.7, 0.3, 0.015, 100] | 0.405              | 28.0          |
| [0.8, 0.2, 0.015, 100] | 0.406              | 27.0          |
| [0.9, 0.1, 0.015, 100] | 0.405              | 25.0          |
| [1.0, 0.0, 0.015, 100] | 0.405              | 25.0          |

- CoreInfer got **19.2**
- Then, I have picked the 6th configuration (sparsity = 0.405) and run it on the whole dataset and got a score of **21.2** (110.4%)
- Then I used the AVG, which has actual sparsity = 0.402 and run it on the whole dataset and got a score of 


### CommonSenseQA:

- CoreInfer actual sparsity for all of them was 0.32
- When determining these sparsity levels, the goal sparsity was set to 0.32:

|  Params                | Actual Sparsity    | Accuracy      |  
|------------------------|:------------------:|:-------------:|
| [0.4, 0.6, 0.02, 100]  | 0.320              | 0.26          |
| [0.5, 0.5, 0.02, 100]  | 0.318              | 0.26          |            
| [0.6, 0.4, 0.02, 100]  | 0.315              | 0.26          |
| [0.7, 0.3, 0.02, 100]  | 0.310              | 0.26          |
| [0.9, 0.1, 0.015, 100] | 0.322              | 0.26          |
| [1.0, 0.0, 0.015, 100] | 0.316              | 0.26          |

- CoreInfer got **0.198**
- Then, I have picked the 1st configuration (sparsity = 0.32) and run it on the whole dataset and got a score of **0.19** (96%)
- Then I used the AVG, which has actual sparsity = 0.318 and run it on the whole dataset and got a score of **0.19** (96%)
