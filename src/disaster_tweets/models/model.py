import whisk
from tensorflow.keras.models import load_model
import keras
from keras.preprocessing.sequence import pad_sequences
import pickle

MAX_LEN=50

class Model:
    def __init__(self):
        self.model = load_model(whisk.artifacts_dir / "model.h5")
        with open(whisk.artifacts_dir / 'tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def predict(self,texts):
        """
        Returns model predictions.
        """
        sequences = self.tokenizer.texts_to_sequences(texts)
        processed_texts = pad_sequences(sequences,maxlen=MAX_LEN,truncating='post',padding='post')
        result = self.model.predict(processed_texts)
        # Calling `tolist()` so the Flask app just works.
        # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        return result.tolist()
