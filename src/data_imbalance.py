# File: src/data_imbalance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DiabetesImbalanceHandler:
    def __init__(self, df, target_col='Outcome', test_size=0.2, random_state=42):
        self.df = df.copy()
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.imbalance_report = {}
        
        # Prepare features and target
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.y  # Maintain class distribution
        )
        
    def analyze_class_distribution(self):
        """Comprehensive class distribution analysis"""
        print("📊 COMPREHENSIVE CLASS IMBALANCE ANALYSIS")
        print("=" * 50)
        
        # Overall distribution
        overall_dist = Counter(self.y)
        train_dist = Counter(self.y_train)
        test_dist = Counter(self.y_test)
        
        print("🎯 CLASS DISTRIBUTION:")
        print(f"Overall Dataset:")
        for class_val, count in overall_dist.items():
            percentage = (count / len(self.y)) * 100
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  - {class_label}: {count} cases ({percentage:.1f}%)")
        
        print(f"\nTraining Set ({len(self.y_train)} samples):")
        for class_val, count in train_dist.items():
            percentage = (count / len(self.y_train)) * 100
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  - {class_label}: {count} cases ({percentage:.1f}%)")
        
        print(f"\nTest Set ({len(self.y_test)} samples):")
        for class_val, count in test_dist.items():
            percentage = (count / len(self.y_test)) * 100
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  - {class_label}: {count} cases ({percentage:.1f}%)")
        
        # Imbalance metrics
        minority_class = min(overall_dist.values())
        majority_class = max(overall_dist.values())
        imbalance_ratio = minority_class / majority_class
        
        print(f"\n⚖️ IMBALANCE METRICS:")
        print(f"  - Imbalance Ratio: {imbalance_ratio:.3f}")
        print(f"  - Minority Class: {minority_class} samples")
        print(f"  - Majority Class: {majority_class} samples")
        
        # Visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Overall distribution
        overall_counts = [overall_dist[0], overall_dist[1]]
        axes[0].pie(overall_counts, labels=['Non-Diabetic', 'Diabetic'], 
                   autopct='%1.1f%%', colors=['lightblue', 'salmon'])
        axes[0].set_title('Overall Class Distribution', fontweight='bold')
        
        # Training distribution
        train_counts = [train_dist[0], train_dist[1]]
        axes[1].pie(train_counts, labels=['Non-Diabetic', 'Diabetic'], 
                   autopct='%1.1f%%', colors=['lightblue', 'salmon'])
        axes[1].set_title('Training Set Distribution', fontweight='bold')
        
        # Test distribution
        test_counts = [test_dist[0], test_dist[1]]
        axes[2].pie(test_counts, labels=['Non-Diabetic', 'Diabetic'], 
                   autopct='%1.1f%%', colors=['lightblue', 'salmon'])
        axes[2].set_title('Test Set Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        self.imbalance_report['class_distribution'] = {
            'overall': overall_dist,
            'train': train_dist,
            'test': test_dist,
            'imbalance_ratio': imbalance_ratio,
            'minority_class': minority_class,
            'majority_class': majority_class
        }
        
        return imbalance_ratio
    
    def evaluate_imbalance_impact(self):
        """Evaluate the impact of imbalance on model performance"""
        print("\n🔍 EVALUATING IMBALANCE IMPACT ON BASELINE MODELS")
        print("=" * 50)
        
        # Scale features for better model performance
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        models = {
            'Random Forest': RandomForestClassifier(random_state=self.random_state),
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }
        
        baseline_results = {}
        
        for name, model in models.items():
            print(f"\n📈 {name} Baseline Performance:")
            
            # Train and predict
            model.fit(X_train_scaled, self.y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = np.mean(y_pred == self.y_test)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Class-wise metrics
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn)  # Recall for positive class
            specificity = tn / (tn + fp)  # Recall for negative class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"  - Accuracy: {accuracy:.3f}")
            print(f"  - AUC-ROC: {auc_roc:.3f}")
            print(f"  - Sensitivity (Diabetic): {sensitivity:.3f}")
            print(f"  - Specificity (Non-Diabetic): {specificity:.3f}")
            print(f"  - Precision (Diabetic): {precision:.3f}")
            
            baseline_results[name] = {
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'confusion_matrix': cm
            }
            
            # Show confusion matrix
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Diabetic', 'Diabetic'],
                       yticklabels=['Non-Diabetic', 'Diabetic'])
            plt.title(f'{name} - Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.show()
        
        self.imbalance_report['baseline_performance'] = baseline_results
        return baseline_results
    
    def apply_sampling_techniques(self):
        """Apply various sampling techniques to handle imbalance"""
        print("\n🔄 APPLYING SAMPLING TECHNIQUES FOR IMBALANCE HANDLING")
        print("=" * 50)
        
        sampling_methods = {
            'Original': None,
            'Random Oversampling': RandomOverSampler(random_state=self.random_state),
            'Random Undersampling': RandomUnderSampler(random_state=self.random_state),
            'SMOTE': SMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'SMOTE + TomekLinks': SMOTETomek(random_state=self.random_state),
            'SMOTE + ENN': SMOTEENN(random_state=self.random_state)
        }
        
        sampling_results = {}
        
        for method_name, sampler in sampling_methods.items():
            print(f"\n🎯 {method_name}:")
            
            if sampler is None:
                # Use original data
                X_resampled, y_resampled = self.X_train, self.y_train
            else:
                # Apply sampling
                X_resampled, y_resampled = sampler.fit_resample(self.X_train, self.y_train)
            
            # Analyze resampled distribution
            resampled_dist = Counter(y_resampled)
            print(f"  - Resampled distribution: {dict(resampled_dist)}")
            
            # Scale features
            scaler = StandardScaler()
            X_resampled_scaled = scaler.fit_transform(X_resampled)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Train Random Forest on resampled data
            rf = RandomForestClassifier(random_state=self.random_state)
            rf.fit(X_resampled_scaled, y_resampled)
            
            # Evaluate
            y_pred = rf.predict(X_test_scaled)
            y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
            
            accuracy = np.mean(y_pred == self.y_test)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            
            cm = confusion_matrix(self.y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            print(f"  - Accuracy: {accuracy:.3f}")
            print(f"  - AUC-ROC: {auc_roc:.3f}")
            print(f"  - Sensitivity: {sensitivity:.3f}")
            print(f"  - Specificity: {specificity:.3f}")
            
            sampling_results[method_name] = {
                'X_resampled': X_resampled,
                'y_resampled': y_resampled,
                'distribution': resampled_dist,
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'confusion_matrix': cm
            }
        
        # Compare sampling techniques
        self._compare_sampling_techniques(sampling_results)
        
        self.imbalance_report['sampling_results'] = sampling_results
        return sampling_results
    
    def _compare_sampling_techniques(self, sampling_results):
        """Compare different sampling techniques"""
        print("\n📊 COMPARISON OF SAMPLING TECHNIQUES")
        print("=" * 50)
        
        comparison_data = []
        for method, results in sampling_results.items():
            comparison_data.append({
                'Method': method,
                'Accuracy': results['accuracy'],
                'AUC-ROC': results['auc_roc'],
                'Sensitivity': results['sensitivity'],
                'Specificity': results['specificity'],
                'Training Samples': len(results['y_resampled'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('AUC-ROC', ascending=False)
        
        print(comparison_df.round(3))
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # AUC-ROC Comparison
        methods = comparison_df['Method']
        auc_scores = comparison_df['AUC-ROC']
        axes[0,0].barh(methods, auc_scores, color='skyblue')
        axes[0,0].set_xlabel('AUC-ROC Score')
        axes[0,0].set_title('AUC-ROC by Sampling Method')
        axes[0,0].set_xlim(0, 1)
        
        # Sensitivity vs Specificity
        sensitivity = comparison_df['Sensitivity']
        specificity = comparison_df['Specificity']
        axes[0,1].scatter(sensitivity, specificity, s=100)
        for i, method in enumerate(methods):
            axes[0,1].annotate(method, (sensitivity[i], specificity[i]), 
                             xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0,1].set_xlabel('Sensitivity (Diabetic Recall)')
        axes[0,1].set_ylabel('Specificity (Non-Diabetic Recall)')
        axes[0,1].set_title('Sensitivity vs Specificity Trade-off')
        axes[0,1].grid(True, alpha=0.3)
        
        # Training Samples
        samples = comparison_df['Training Samples']
        axes[1,0].barh(methods, samples, color='lightgreen')
        axes[1,0].set_xlabel('Training Samples')
        axes[1,0].set_title('Training Set Size After Sampling')
        
        # Accuracy Comparison
        accuracy = comparison_df['Accuracy']
        axes[1,1].barh(methods, accuracy, color='lightcoral')
        axes[1,1].set_xlabel('Accuracy')
        axes[1,1].set_title('Accuracy by Sampling Method')
        axes[1,1].set_xlim(0, 1)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def select_optimal_sampling_method(self, sampling_results):
        """Select the optimal sampling method based on comprehensive evaluation"""
        print("\n🎯 SELECTING OPTIMAL SAMPLING METHOD")
        print("=" * 50)
        
        # Define scoring weights (prioritize sensitivity for medical context)
        weights = {
            'auc_roc': 0.3,
            'sensitivity': 0.4,  # Higher weight for detecting diabetics
            'specificity': 0.2,
            'accuracy': 0.1
        }
        
        method_scores = {}
        
        for method, results in sampling_results.items():
            score = (
                results['auc_roc'] * weights['auc_roc'] +
                results['sensitivity'] * weights['sensitivity'] +
                results['specificity'] * weights['specificity'] +
                results['accuracy'] * weights['accuracy']
            )
            method_scores[method] = score
        
        # Select best method
        best_method = max(method_scores, key=method_scores.get)
        best_results = sampling_results[best_method]
        
        print(f"🏆 OPTIMAL METHOD: {best_method}")
        print(f"📊 Composite Score: {method_scores[best_method]:.3f}")
        print(f"📈 Key Metrics:")
        print(f"  - AUC-ROC: {best_results['auc_roc']:.3f}")
        print(f"  - Sensitivity: {best_results['sensitivity']:.3f}")
        print(f"  - Specificity: {best_results['specificity']:.3f}")
        print(f"  - Accuracy: {best_results['accuracy']:.3f}")
        
        # Create final balanced dataset
        if best_method == 'Original':
            X_final, y_final = self.X_train, self.y_train
        else:
            X_final, y_final = best_results['X_resampled'], best_results['y_resampled']
        
        final_distribution = Counter(y_final)
        print(f"\n📦 FINAL BALANCED DATASET:")
        print(f"  - Samples: {len(X_final)}")
        print(f"  - Distribution: {dict(final_distribution)}")
        
        self.imbalance_report['optimal_method'] = {
            'method': best_method,
            'score': method_scores[best_method],
            'X_balanced': X_final,
            'y_balanced': y_final,
            'distribution': final_distribution,
            'all_scores': method_scores
        }
        
        return X_final, y_final, best_method
    
    def create_balanced_dataset(self, X_balanced, y_balanced):
        """Create final balanced dataset for modeling"""
        print("\n💾 CREATING FINAL BALANCED DATASET")
        print("=" * 50)
        
        # Combine features and target
        balanced_df = pd.DataFrame(X_balanced, columns=self.X.columns)
        balanced_df[self.target_col] = y_balanced
        
        # Add original test set for final evaluation
        test_df = pd.DataFrame(self.X_test, columns=self.X.columns)
        test_df[self.target_col] = self.y_test
        
        print("Final Dataset Summary:")
        print(f"📊 Balanced Training Set: {balanced_df.shape}")
        print(f"🎯 Test Set: {test_df.shape}")
        
        # Show final distribution
        balanced_dist = Counter(balanced_df[self.target_col])
        test_dist = Counter(test_df[self.target_col])
        
        print(f"\n📈 Final Class Distribution:")
        print(f"Balanced Training:")
        for class_val, count in balanced_dist.items():
            percentage = (count / len(balanced_df)) * 100
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  - {class_label}: {count} cases ({percentage:.1f}%)")
        
        print(f"Test Set (Original Distribution):")
        for class_val, count in test_dist.items():
            percentage = (count / len(test_df)) * 100
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  - {class_label}: {count} cases ({percentage:.1f}%)")
        
        self.imbalance_report['final_datasets'] = {
            'balanced_train': balanced_df,
            'test_set': test_df
        }
        
        return balanced_df, test_df
    
    def generate_imbalance_summary(self):
        """Generate comprehensive imbalance handling summary"""
        print("\n📋 DATA IMBALANCE HANDLING SUMMARY REPORT")
        print("=" * 60)
        
        print("🚀 IMBALANCE HANDLING OPERATIONS PERFORMED:")
        print("• Comprehensive class distribution analysis")
        print("• Baseline model performance evaluation")
        print("• Multiple sampling techniques application")
        print("• Optimal method selection")
        print("• Final balanced dataset creation")
        
        print("\n📊 KEY ACHIEVEMENTS:")
        if 'class_distribution' in self.imbalance_report:
            orig_ratio = self.imbalance_report['class_distribution']['imbalance_ratio']
            print(f"• Original imbalance ratio: {orig_ratio:.3f}")
        
        if 'optimal_method' in self.imbalance_report:
            best_method = self.imbalance_report['optimal_method']['method']
            final_dist = self.imbalance_report['optimal_method']['distribution']
            print(f"• Optimal sampling method: {best_method}")
            print(f"• Final training distribution: {dict(final_dist)}")
        
        if 'final_datasets' in self.imbalance_report:
            balanced_shape = self.imbalance_report['final_datasets']['balanced_train'].shape
            test_shape = self.imbalance_report['final_datasets']['test_set'].shape
            print(f"• Balanced training set: {balanced_shape}")
            print(f"• Test set: {test_shape}")
        
        print("\n💡 RECOMMENDATIONS FOR MODEL TRAINING:")
        print("• Use balanced dataset for training")
        print("• Evaluate on original test distribution")
        print("• Focus on sensitivity and AUC-ROC metrics")
        print("• Consider clinical implications of predictions")
        
        return self.imbalance_report

# Usage function
def execute_imbalance_handling_pipeline(df, target_col='Outcome'):
    """Execute complete data imbalance handling pipeline"""
    handler = DiabetesImbalanceHandler(df, target_col)
    
    # Step 1: Analyze class distribution
    imbalance_ratio = handler.analyze_class_distribution()
    
    # Step 2: Evaluate baseline performance
    baseline_results = handler.evaluate_imbalance_impact()
    
    # Step 3: Apply sampling techniques
    sampling_results = handler.apply_sampling_techniques()
    
    # Step 4: Select optimal method
    X_balanced, y_balanced, best_method = handler.select_optimal_sampling_method(sampling_results)
    
    # Step 5: Create final datasets
    balanced_df, test_df = handler.create_balanced_dataset(X_balanced, y_balanced)
    
    # Step 6: Generate summary
    imbalance_report = handler.generate_imbalance_summary()
    
    return balanced_df, test_df, imbalance_report