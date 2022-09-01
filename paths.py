from pathlib import Path
from dotenv import dotenv_values

# Path to .env file
config = dotenv_values(".env")

# Data folder
DATA = Path(config['ROOT']) / 'data'
OUTPUT = Path(config['ROOT']) / 'output'
CONFIG = Path(config['ROOT']) / 'configs'

# -> Data manipulation
RAW         = DATA / 'raw'
PROCESSED   = DATA / 'processed'

# -> Data -> processed 
GRAPHS          = PROCESSED / 'graphs'
TRAINING_MODELS = PROCESSED / 'models'
CHECKPOINT      = PROCESSED / 'checkpoint'

# -> Output
IMGS        = OUTPUT / 'imgs'
RESULTS     = OUTPUT / 'results'
MATRIX     = OUTPUT / 'matrix'


