# sentiment-analysis
## Installation
1. Clone the repository
<br/>`git clone https://github.com/Ruin9999/sentiment-analysis.git`
<br/>`cd sentiment-analysis`
3. Create virtual environment
<br/>**For Unix/macOS**
<br/>`python -m venv venv`
<br/>`source venv/bin/activate`
<br/>**For Windows**
<br/>`py -m venv venv`
<br/>`.\venv\Scripts\activate`
4. Execute `pip install -r requirements.txt` to install all required packages.


## Running
Using RNN as an example (Default is UnFreeze):
* Run inference and comparison with Best Models - TODO
<br/>`python sentiment-analysis/main.py`

* Run Tuning and Training
<br/>`python sentiment-analysis/tuning.py` --model rnn
    Defualt n_trials=15
    You can specify the parameters tuning in /sentiment-analysis/training/hyperparameter_tuning.py for each model

* Best Model are saved in /checkpoints/final_model

* Best Config are stored in /configs/hyperparam_tuning

* Optional Model: 
    'rnn': SentimentRNN,
    'bilstm': SentimentBiLSTM,
    'bigru': SentimentBiGRU,
    'cnn': ImprovedSentimentCNN,
