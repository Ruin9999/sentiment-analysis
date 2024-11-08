# sentiment-analysis
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
Using RNN as an example (Default is UnFreeze):

* Run comparison and sample inference with Best Models - TO Debug
  `python sentiment-analysis/main.py`

* Run Tuning and Training  
  `python sentiment-analysis/tuning.py --model rnn`  
  Default n_trials=15  
  You can specify the parameters tuning in `/sentiment-analysis/training/hyperparameter_tuning.py` for each model

* Best Model are saved in Google Drive due to the Git limit.  
   Please Download from `https://drive.google.com/drive/folders/1S5Fm44GBtja50LvdmBsBSvlsdsy5g947?usp=drive_link`  
   Then copy the .ckpt checkpoints to `best_models/final_model`

* Best Config are stored in `best_models/final_config`

* Optional Models:  
  'rnn': SentimentRNN  
  'rnn_freeze': SentimentRNN  
  'bilstm': SentimentBiLSTM  
  'bigru': SentimentBiGRU  
  'cnn': ImprovedSentimentCNN  

