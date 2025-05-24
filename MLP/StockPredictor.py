import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class BTCMLPClassifier: 
    def __init__(self):
        self.model = None
        self.history = None
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        self.target_column = 'Output'
        
    def load_data(self, train_path='./data/BTC_train.csv', test_path='./data/BTC_test.csv'):
        try:
            self.train_df = pd.read_csv(train_path)
            self.test_df = pd.read_csv(test_path)
            
            X_train = self.train_df[self.feature_columns].values
            y_train = self.train_df[self.target_column].values
            X_test = self.test_df[self.feature_columns].values
            y_test = self.test_df[self.target_column].values
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise e
    
    def build_model(self, input_dim, architecture=[128, 64, 32], dropout_rate=0.3, l2_reg=0.001, learning_rate=0.001):
        self.model = Sequential()
        
        # Input layer
        self.model.add(Dense(architecture[0], 
                           input_dim=input_dim,
                           activation='relu',
                           kernel_regularizer=l2(l2_reg)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in architecture[1:]:
            self.model.add(Dense(units, 
                               activation='relu',
                               kernel_regularizer=l2(l2_reg)))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(dropout_rate))
        
        # Output layer
        self.model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
    def train_model(self, X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, patience=15):
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=0
            )
        ]
        
        # class weights
        class_counts = np.bincount(y_train.astype(int))
        total_samples = len(y_train)
        class_weight = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
    def evaluate_model(self, X_train, y_train, X_test, y_test, threshold=0.5):
        # Training
        y_train_pred = (self.model.predict(X_train, verbose=0) > threshold).astype(int).flatten()
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        
        # Testing
        y_test_pred = (self.model.predict(X_test, verbose=0) > threshold).astype(int).flatten()
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        return train_accuracy, train_f1, test_accuracy, test_f1


def main():
    classifier = BTCMLPClassifier()
    X_train, y_train, X_test, y_test = classifier.load_data()
    
    model_path = 'MLP/trained_btc_model.keras'
    
    if os.path.exists(model_path):
        print("Using existing trained model")
        classifier.model = load_model(model_path)
    else:
        print("Training new model")

        classifier.build_model(input_dim=X_train.shape[1])
        classifier.train_model(X_train, y_train)
        
        classifier.model.save(model_path)
    
    train_acc, train_f1, test_acc, test_f1 = classifier.evaluate_model(X_train, y_train, X_test, y_test)
    
    # Output results
    print(f"Seed: {SEED}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Training F1: {train_f1:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Testing F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()