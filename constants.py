import os


PROJECT_DIR = os.path.dirname('__file__')

CLASS_LIST = [
    'Dostoevsky', 'Tolstoy', 'Turgenev',
    'Chekhov', 'Lermontov', 'Pushkin'
]

EMBEDDING_DIM = 300
FILTERS = (
    '!"#$%&()*+,-–—./…:;<=>?@[\\]^_`{|}~«»\t\n\xa0\ufeff'
)
MAX_WORDS = 10000
MAX_SEQ = 40000

WIN_SIZE = 1000
WIN_STEP = 100

BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.1
