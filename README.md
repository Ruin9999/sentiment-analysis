# sentiment-analysis

## Code Structure  
```plaintext
sentiment-analysis/  
├── best_models  
│   └── final_config  
│       ├── ImprovedSentimentCNN_final_config.json  
│       ├── SentimentBiGRU_final_config.json  
│       ├── SentimentBiLSTM_final_config.json  
│       ├── SentimentRNN_final_config.json  
│       └── SentimentRNN-Freeze_final_config.json  
├── original_notebooks  
│   ├── sentiment_analysis_part1.ipynb  
│   ├── sentiment_analysis_part2.ipynb  
│   ├── sentiment_analysis_part3_1_2.ipynb  
│   ├── sentiment_analysis_part3.3.ipynb  
│   ├── sentiment_analysis_part_3_4.ipynb   
|   └── final_improve_bert.py  
├── README.md   
├── requirements.txt  
└── sentiment-analysis  
    ├── data  
    │   ├── data_loader.py  
    │   └── preprocessing.py  
    ├── embeddings  
    │   └── embedding_preparation.py  
    ├── main.py  
    ├── models  
    │   ├── bigru.py  
    │   ├── bilstm.py  
    │   ├── cnn.py  
    │   ├── improved_cnn.py  
    │   └── rnn.py  
    ├── training  
    │   ├── hyperparameter_tuning.py  
    │   └── trainer.py  
    ├── tuning.py  
    └── utils  
        ├── get_model.py  
        ├── metrics.py  
        └── oov_handler.py 
```

## Installation

**Prerequisite**: Python version >= 3.9

1. Clone the repository  
   `git clone https://github.com/Ruin9999/sentiment-analysis.git`  
   `cd sentiment-analysis`

2. Create virtual environment  
   **For Unix/macOS**  
   `python -m venv venv`  
   `source venv/bin/activate`  
   **For Windows**  
   `py -m venv venv`  
   `.\venv\Scripts\activate`

3. Ensure pip is up to date before installing the required packages   
   `python -m pip install --upgrade pip`   

4. Execute `pip install -r requirements.txt` to install all required packages 

## Running  

* Code for all 3 parts are saved in jupyter notebooks.  
   Existing output are stored together with the notebooks.
   You can run each notebook in `/original_notebooks`  

* Train and Run Bert model for part 3.5 solution   
  `python original_notebooks/final_improve_bert.py`   
  ```python
   # The default config is listed as below. Please change based on your own device.
   # Training was done on 4 * A100 40G GPU. 
   CONFIG = {
      "model_name": "bert-base-uncased",
      "batch_size": 256,
      "max_length": 128,
      "epochs": 30,
      "learning_rate": 2e-5,
      "weight_decay": 0.01,
      "logging_steps": 10,
      "evaluation_strategy": "epoch",
      "save_strategy": "epoch",
   }
   ```  

* Run Tuning and Training for the tested models  
  `python sentiment-analysis/tuning.py --model rnn`  
  Default `n_trials=15`  
  You can specify the parameters tuning in `/sentiment-analysis/training/hyperparameter_tuning.py` for each model

* Optional Models:  
  'rnn': SentimentRNN  
  'rnn_freeze': SentimentRNN  
  'bilstm': SentimentBiLSTM  
  'bigru': SentimentBiGRU  
  'cnn': ImprovedSentimentCNN  

* Best Model are saved in Google Drive due to the Git limit.  
   Please Download from `https://drive.google.com/drive/folders/1S5Fm44GBtja50LvdmBsBSvlsdsy5g947?usp=drive_link`  
   Then copy the .ckpt checkpoints to directory `best_models/final_model`  

* Best Config are stored in `best_models/final_config`

* Run comparison and sample inference with Best Models
  `python sentiment-analysis/main.py`