import os


# Root directory of the project
PROJECT_DIR: str = os.path.dirname(__file__)

# List of target classes (authors)
CLASS_LIST = [
    'Dostoevsky', 'Tolstoy', 'Turgenev',
    'Chekhov', 'Lermontov', 'Pushkin'
]

# Setting for text proccesing
EMBEDDING_DIM: int = 300
FILTERS: str = (
    '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
)
MAX_WORDS: int = 10000
MAX_SEQ: int = 40000

# Sliding window parameters
WIN_SIZE: int = 1000
WIN_STEP: int = 100

# Training configuration
BATCH_SIZE: int = 64
EPOCHS: int = 50
VALIDATION_SPLIT: float = 0.1
