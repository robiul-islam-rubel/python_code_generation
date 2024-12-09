
# Python Code Generation Using Pre-trained Models

Python code generation involves the use of pre-trained models to generate syntactically and semantically correct Python code based on prompts or input descriptions. Pre-trained models are typically large-scale language models fine-tuned on programming languages (e.g., Python, Java, C++) and are capable of understanding natural language and producing functional code.

###  Benefits of Using Pre-trained Models
Efficiency: Speeds up the development process by automating repetitive coding tasks.
Versatility: Can generate code for a wide range of tasks, from simple functions to complex scripts.
Learning Aid: Assists developers in understanding best practices and common programming patterns.
Error Reduction: Helps identify and resolve syntax or logical errors in code.

#### Pipeline Description

To replicate the experiments you can rely on this two files *vulrepair_main.py* and *vulrepair_main_prompt.py*.
While the former can be used to retrain the original VulRepair approach as well as the ablation model (i.e., T5-base without pre-training), the latter serves the promp-tuning procedure.  Before starting replicating any of the experiments we performed, make sure to install the requirements (see *requirements.txt*)

#### Data Statistics

Data statistics of the datasets are shown in the below table:

|         | #Examples   | #Examples       | #Examples
| ------- | :-------:   | :-------:       | :-------:
|         |   BLEU Score |  ROUGE-L  | BERT Score
|  CodeLlama-7b  |   30.78     |    0.38       |  0.87
|  CodeT5-Large  |   14.56      |    0.31          |  0.59
|   GPT-2  |  18.83     |    0.28        |  0.85

#### Fine-tuning  

*The following starts a fine-tuning procedure using the classic supervised approach*

##### Training




#### Datasets :paperclip:

* The datasets for fine-tuning the models are stored on GDrive <a href="https://drive.google.com/drive/folders/1-3eLMTVLx8evwC9ROUBq9q-IdYHuy_WK?usp=sharing">here</a>
* The dataset for supervised pre-training (Chen et al.) is available here <a href="https://drive.google.com/drive/folders/1DtaNb2FaxGiei8DI1NGy4X9tYlYsxtp3?usp=sharing">here</a>
