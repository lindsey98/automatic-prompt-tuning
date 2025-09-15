
## Installation

To install, simply run:

```commandline
pip install pandas tqdm openai
```

And add your OPENAI_API_KEY with the following command:

```commandline
export OPENAI_API_KEY=YOUR_KEY
```

## Using `APT`

- ```prompts/fatty_liver_init_prompt.txt```: initial prompt seed for fatty liver
- ```prompts/elevated_fbg_init_prompt.txt```: initial prompt seed for FBG

### Run prompt tuning for predicting fatty liver
Run
```commandline
python predict_prompt.py
```

the predicted results will be saved under ``outputs/``

### Run prompt tuning for predicting FBG
Run
```commandline
python predict_prompt_diabete.py
```

