from tensorflow.keras import models, layers

from constants import CLASS_LIST, EMBEDDING_DIM, MAX_WORDS, WIN_SIZE
from dataset import embedding_matrix


def build_model():
    """Build and return the text classification neural network."""
    return models.Sequential([
        layers.Embedding(
            MAX_WORDS, 
            EMBEDDING_DIM,
            input_length=WIN_SIZE,
            weights=[embedding_matrix]
        ),
        
        layers.BatchNormalization(),
        
        layers.Dense(40, activation='relu'),
        
        layers.Dropout(0.6),
        
        layers.BatchNormalization(),
        
        layers.Flatten(),
        
        layers.Dense(len(CLASS_LIST), activation='softmax')
    ])
