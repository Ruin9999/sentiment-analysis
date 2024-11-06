# sentiment-analysis
## Installation
1. Clone the repository  
   `git clone https://github.com/Ruin9999/sentiment-analysis.git`  
   `cd sentiment-analysis`

3. Create virtual environment  
   **For Unix/macOS**  
   `python -m venv venv`  
   `source venv/bin/activate`  
   **For Windows**  
   `py -m venv venv`  
   `.\venv\Scripts\activate`

4. Execute `pip install -r requirements.txt` to install all required packages.

## Running
Using RNN as an example (Default is UnFreeze):

* Run inference and comparison with Best Models - TODO  
  `python sentiment-analysis/main.py`

* Run Tuning and Training  
  `python sentiment-analysis/tuning.py --model rnn`  
  Default n_trials=15  
  You can specify the parameters tuning in `/sentiment-analysis/training/hyperparameter_tuning.py` for each model

* Best Model are saved in `/checkpoints/final_model`

* Best Config are stored in `/configs/hyperparam_tuning`

* Optional Models:  
  `'rnn': SentimentRNN,  
  'bilstm': SentimentBiLSTM,  
  'bigru': SentimentBiGRU,  
  'cnn': ImprovedSentimentCNN`
