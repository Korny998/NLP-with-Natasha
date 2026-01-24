import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def show_plot(history):
    """Plot training and validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle('Graph of the model learning process')
    ax1.plot(
        history.history['accuracy'],
        label='Accuracy graph on the training sample'
    )
    ax1.plot(
        history.history['val_accuracy'],
        label='Graph of accuracy in the test sample'
    )
    ax1.xaxis.get_major_locator().set_params(integer=True)
    ax1.set_xlabel('The age of learning')
    ax1.set_ylabel('Accuracy graph')
    ax1.legend()

    ax2.plot(
        history.history['loss'],
        label='Error in the training sample'
    )
    ax2.plot(
        history.history['val_loss'],
        label='Error in the test sample'
    )
    ax2.xaxis.get_major_locator().set_params(integer=True)
    ax2.set_xlabel('The age of learning')
    ax2.set_ylabel('Mistake')
    ax2.legend()
    plt.show()


def show_confusion_matrix(y_true, y_pred, class_labels):
    """Display a normalized confusion matrix."""
    cm = confusion_matrix(
        np.argmax(y_true, axis=1),
        np.argmax(y_pred, axis=1),
        normalize='true'
    )
    cm = np.around(cm, 3)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Error matrix', fontsize=18)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )
    disp.plot(ax=ax)
    plt.gca().images[-1].colorbar.remove()
    plt.xlabel('', fontsize=16)
    plt.ylabel('', fontsize=16)
    fig.autofmt_xdate(rotation=45)
    plt.show()
