# Explaining `coreinfer.py`:

This will explain the flow of running inference using the `llama` model. 

## `main` function:
- First it calls `_load_model`.
- Then it calls `convert_model`.
- Finally it calls `generate`.


## `_load_model` function:
- This function basically loads the entire model into the selected `device`.
- If `memory_limit` is set to true, then it calls the `load_llama_model` function.
    ### `load_llama_model` function:
    - It basically loads the entire model to the GPU except some params.
    - For each layer between `start_num` and `end_num`, it moves the weights of the Gate, Up and Down matrices to the CPU in half precision format.

## `convert_model` function:
- It will call the relevant convert function based on the model and the method.
- If the method is stable, it will call `convert_llama_model`.
    ### `convert_llama_model` function:
    - For each layer between `start_num` and `end_num`, it replaces the models Gate, Up and Down classes with custom classes.
    - For more details on these 3 classes, check the comments in the code.
- If the method is similarity, it will call `convert_llama_model_sim`

## `generate` function:
- First, we prepare the prompt by appending some pre-defined text to the prompt based on the task. This is done to help stablize the prompt and have more clear semantics, so that the core neurons can be correct.
- Then we generate the next token using the input prompt, and we save the `past_key_values`. This stores attention states to avoid redundant computations in future tokens.
- Then we loop `num_tokens_to_generate` times, and we generate new tokens.
- They are using the `argmax` sampling, which is a greedy method that always selects the token with the highest probability.