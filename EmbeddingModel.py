from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Embedding
import flag


class EmbeddingModel:

    def __init__(self, word_index):
        self._word_index = word_index

    def build_model(self):
        self.model = Sequential()
        # Embedding model
        self.model.add(Embedding(len(self._word_index), output_dim=flag.embedding_dim, input_length=flag.sequence_length))
        self.model.compile('')
        return self.model