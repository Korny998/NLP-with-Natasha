from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from constants import BATCH_SIZE, EPOCHS, VALIDATION_SPLIT
from dataset import x_train, x_test, y_train
from model import build_model


def train_model():
    model = build_model()
    
    model.layers[0].trainable = False
    
    model.compile(
        optimizer=Adam(),
        loss=CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    history_train = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    model.save_weights('pre_trained_model.h5')
    
    return model, history


if __name__ == '__main__':
    model_train, history_train = train_model()
    y_pred = model_train.predict(x_test)
