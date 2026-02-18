import os
import zipfile

import glob
import numpy as np
from navec import Navec
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer

from constants import (
    CLASS_LIST, EMBEDDING_DIM, FILTERS,
    MAX_WORDS, MAX_SEQ, PROJECT_DIR,
    WIN_SIZE, WIN_STEP
)


navec = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')

data_path = utils.get_file(
    'russian_literature.zip',
    'https://storage.yandexcloud.net/academy.ai/russian_literature.zip'
)

data_dir: str = os.path.join(PROJECT_DIR, 'dataset')
os.makedirs(data_dir, exist_ok=True)

if not os.listdir(data_dir):
    with zipfile.ZipFile(data_path, 'r') as z:
        z.extractall(data_dir)

all_texts: dict = {}

for author in CLASS_LIST:
    all_texts[author] = ''
    for path in (
        glob.glob(os.path.join(data_dir, 'prose', author, '*.txt')) +
        glob.glob(os.path.join(data_dir, 'poems', author, '*.txt'))
    ):
        with open(path, 'r', errors='ignore') as file:
            text = file.read()
            all_texts[author] += text.replace('\n', ' ')

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    filters=FILTERS,
    lower=True,
    split=' ',
    char_level=False
)

tokenizer.fit_on_texts(all_texts.values())

seq_train = tokenizer.texts_to_sequences(all_texts.values())

seq_train_balance = [seq_train[cls][:MAX_SEQ] for cls in range(len(CLASS_LIST))]


def seq_split(sequence, win_size: int, win_step: int) -> list:
    """Split a token sequence into overlapping fixed-length windows."""
    return [
        sequence[i:i + win_size] for i in range(
            0, len(sequence) - win_size + 1, win_step
        )
    ]


def seq_vectorize(
    seq_list,
    test_split,
    class_list,
    win_size,
    step
):
    """Convert tokenized text sequences into training and testing datasets."""
    assert len(seq_list) == len(class_list)

    x_train, y_train, x_test, y_test = [], [], [], []
    num_classes = len(class_list)

    for cls in range(num_classes):
        sequence = seq_list[cls]
        gate = int(len(sequence) * (1 - test_split))

        train_window = seq_split(sequence[:gate], win_size, step)
        test_window = seq_split(sequence[gate:], win_size, step)

        if not train_window:
            continue

        x_train.extend(train_window)
        x_test.extend(test_window)

        y_train.extend(
            [utils.to_categorical(cls, num_classes)] * len(train_window)
        )
        y_test.extend(
            [utils.to_categorical(cls, num_classes)] * len(test_window)
        )

    return (
        np.array(x_train, dtype=np.int32),
        np.array(y_train, dtype=np.float32),
        np.array(x_test, dtype=np.int32),
        np.array(y_test, dtype=np.float32)
    )


x_train, y_train, x_test, y_test = seq_vectorize(
    seq_train_balance,
    0.1,
    CLASS_LIST,
    WIN_SIZE,
    WIN_STEP
)

word_index = tokenizer.word_index
embedding_index = navec

embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_WORDS:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
