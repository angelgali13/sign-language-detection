import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "sign_language_model.h5")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.npy")
LOG_DIR = os.path.join(BASE_DIR, "logs")
for d in [DATASET_DIR, MODEL_DIR, LOG_DIR]: os.makedirs(d, exist_ok=True)
SIGNS = ["hello","thanks","yes","no","please","help","sorry","name","good","bad"]
NUM_SEQUENCES = 30; SEQUENCE_LENGTH = 30; COLLECTION_WAIT = 30
KEYPOINT_SIZE = 126
LSTM_UNITS = [64,128,64]; DENSE_UNITS = [64,32]; DROPOUT_RATE = 0.3
LEARNING_RATE = 1e-3; EPOCHS = 200; BATCH_SIZE = 32; VALIDATION_SPLIT = 0.2
PREDICTION_THRESHOLD = 0.85; SMOOTHING_WINDOW = 5
