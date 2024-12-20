
# Python Code Generation Using Pre-trained Models

Python code generation involves the use of pre-trained models to generate syntactically and semantically correct Python code based on prompts or input descriptions. Pre-trained models are typically large-scale language models fine-tuned on programming languages (e.g., Python, Java, C++) and are capable of understanding natural language and producing functional code.

###  Benefits of Using Pre-trained Models
Efficiency: Speeds up the development process by automating repetitive coding tasks.<br>
Versatility: Can generate code for a wide range of tasks, from simple functions to complex scripts. <br>
Learning Aid: Assists developers in understanding best practices and common programming patterns.<br>
Error Reduction: Helps identify and resolve syntax or logical errors in code.<be>

## Minimum configuration

- I Only test A40 GPUs with 46GB memory for finetune.
- If you see a ```CUDA``` error, remember to use ```CUDA_VISIBLE_DEVICES=3```, where 3 is the GPU number, you can use any GPU that is available or you can also set the device in the ```set_device(0)```, here 0 is GPU number.


### Example of fine-tuning model data format
```python
 {
    "docstring": "Write a Python function to add two numbers and return the result.",
    "code": "def add_numbers(a, b):\n    result = a + b\n    return result"
}
```

#### Pipeline Description

To replicate the experiments, you can rely on any of these files: ```codellama-7b.py``` and ```gpt2.py```. For fine-tuning, you can run any file that does not start with ``infer```.
Before starting replicating any of the experiments we performed, make sure to install the requirements (see ```requirements.txt```)

#### Evaluation

The evaluation of different models is shown in the table below:

|         | #Examples   | #Examples       | #Examples
| ------- | :-------:   | :-------:       | :-------:
|         |   BLEU Score |  ROUGE-L  | BERT Score
|  CodeLlama-7b  |   30.78     |    0.38       |  0.87
|  CodeT5-Large  |   14.56      |    0.31          |  0.59
|   GPT-2  |  18.83     |    0.28        |  0.85



##### Inference
You can run any file that starts with infer for inference. Remember that to run the Codellama-13b file, you need higher ```GPU``` resources.




#### Datasets

* The dataset (```CodeSearchNet```) for fine-tuning the models are available on <a href="https://huggingface.co/datasets/code-search-net/code_search_net">huggingface</a>
* The dataset (```CodeAlpaca_20k```) for fine-tuning the models are available on <a href="https://huggingface.co/datasets/HuggingFaceH4/CodeAlpaca_20K">huggingface</a>

N.B: You should use ```Python 3.10.12``` for unnecessary Python version problems. <be>
* ```codet5-base.py```, if you want to fine-tune the large one then you just need to change the model name. It is the same for the ```CodeLlama``` as well.
