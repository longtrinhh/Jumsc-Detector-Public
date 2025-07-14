import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import time
import os
from data_collector import create_training_data, extract_audio_features

class JumpscareDetector:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self._init_model()
    
    def _init_model(self):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced data
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced'
            )
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
    
    def train(self, X, y):
        print(f"\nTraining {self.model_type.replace('_', ' ')} model")
        
        # Split data
        print("Splitting data into train/test sets (80/20)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"   Train set: {X_train.shape[0]} samples")
        print(f"   Test set: {X_test.shape[0]} samples")
        
        # Scale features
        print("Scaling features...")
        start_time = time.time()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"   Scaling completed in {time.time() - start_time:.2f}s")
        
        # Train model
        print(f"Training {self.model_type} model...")
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        train_time = time.time() - start_time
        print(f"   Training completed in {train_time:.2f}s")
        
        # Evaluate
        print("Evaluating model performance...")
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nResults for {self.model_type}:")
        print(f"   Train Accuracy: {train_score:.4f}")
        print(f"   Test Accuracy: {test_score:.4f}")
        print(f"   ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        print(f"   Training Time: {train_time:.2f}s")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top_features = np.argsort(importance)[-10:]
            print(f"\n🔝 Top 10 Important Features:")
            for i, idx in enumerate(top_features[::-1]):
                print(f"   {i+1}. Feature {idx}: {importance[idx]:.4f}")
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'roc_auc': roc_auc_score(y_test, y_prob),
            'training_time': train_time,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_jumpscare_probability(self, audio_path, start_time):
        """Predict probability of jumpscare for a 3-second segment"""
        features = extract_audio_features(audio_path, start_time).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0, 1]
        return probability
    
    def analyze_full_audio(self, audio_path, threshold=0.4, min_gap=3.0, step_size=2):
        """Analyze entire audio file and return jumpscare timestamps"""
        import librosa
        
        duration = librosa.get_duration(path=audio_path)
        jumpscares = []
        
        print(f"Analyzing {duration:.1f}s of audio with step size {step_size}s...")
        
        # Analyze every step_size seconds for faster processing
        for start_time in range(0, int(duration - 3), step_size):
            prob = self.predict_jumpscare_probability(audio_path, start_time)
            
            if prob > threshold:
                # Check if this is too close to a previous detection
                if not jumpscares or (start_time - jumpscares[-1]) >= min_gap:
                    jumpscares.append(start_time + 1.5)  # Middle of 3-second window
                    print(f"Potential jumpscare at {start_time + 1.5:.1f}s (confidence: {prob:.3f})")
        
        return jumpscares
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        print(f"Saving model to {filepath}...")
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"   Model saved successfully!")
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")

def hyperparameter_tuning(X, y, model_type='random_forest'):
    """Perform hyperparameter tuning"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if model_type == 'random_forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    elif model_type == 'gradient_boost':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        }
        model = GradientBoostingClassifier(random_state=42)
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters for {model_type}:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Example training pipeline
    print("Jumpscare Detection Model Training")
    
    # Try converted labels file first, then fall back to original
    labels_file = "video_labels_converted.json" if os.path.exists("video_labels_converted.json") else "video_labels.json"
    print(f"Using labels file: {labels_file}")
    
    try:
        # Load and process training data
        X, y = create_training_data(labels_file)
        print(f"\nDataset Summary:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Jumpscare segments: {np.sum(y)} ({np.sum(y)/len(y)*100:.1f}%)")
        print(f"   Normal segments: {len(y) - np.sum(y)} ({(len(y) - np.sum(y))/len(y)*100:.1f}%)")
        
        # Train different models
        models = ['random_forest', 'gradient_boost', 'neural_network']
        results = {}
        
        total_start_time = time.time()
        
        for i, model_type in enumerate(models, 1):
            print(f"\nTraining Model {i}/{len(models)}: {model_type.replace('_', ' ')}")
            
            detector = JumpscareDetector(model_type)
            results[model_type] = detector.train(X, y)
            
            # Save the model
            detector.save_model(f"jumpscare_model_{model_type}.pkl")
        
        total_time = time.time() - total_start_time
        
        # STAGE 3: Compare results
        print(f"\n{'='*60}")
        print("🏆 FINAL MODEL COMPARISON")
        print('='*60)
        
        best_model = None
        best_score = 0
        
        for model_type, result in results.items():
            test_acc = result['test_accuracy']
            roc_auc = result['roc_auc']
            train_time = result['training_time']
            
            print(f"\n{model_type.replace('_', ' ')}:")
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   ROC AUC: {roc_auc:.4f}")
            print(f"   Training Time: {train_time:.2f}s")
            
            if roc_auc > best_score:
                best_score = roc_auc
                best_model = model_type
        
        print(f"\nBest Model: {best_model.replace('_', ' ')}")
        print(f"Best ROC AUC: {best_score:.4f}")
        print(f"Total Training Time: {total_time:.2f}s")
        print(f"\nTraining complete! Models saved as .pkl files")
        print("Ready to use with backend_ml.py")
        
    except FileNotFoundError:
        print(f"Please create {labels_file} with your labeled training data")
        print("Example format:")
        example = {
            "https://youtube.com/watch?v=example1": {
                "jumpscares": [12.5, 45.2],
                "audio_file": "audio/example1.wav"
            },
            "https://youtube.com/watch?v=example2": {
                "jumpscares": [],
                "audio_file": "audio/example2.wav"
            }
        }
        print(json.dumps(example, indent=2)) 