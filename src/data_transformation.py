# File: src/data_transformation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesDataTransformer:
    def __init__(self, df):
        self.df = df.copy()
        self.transformation_report = {}
        
    def create_age_groups(self):
        """Create clinically meaningful age groups"""
        print("🎯 CREATING AGE GROUPS")
        print("=" * 40)
        
        try:
            bins = [20, 30, 40, 50, 60, 100]
            labels = ['20-29', '30-39', '40-49', '50-59', '60+']
            self.df['Age_Group'] = pd.cut(self.df['Age'], bins=bins, labels=labels, right=False)
            
            age_group_counts = self.df['Age_Group'].value_counts().sort_index()
            print("Age Group Distribution:")
            for group, count in age_group_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  - {group}: {count} cases ({percentage:.1f}%)")
            
            self.transformation_report['age_groups'] = age_group_counts
            return self.df
        except KeyError:
            print("⚠️ Column 'Age' not found. Skipping age group creation.")
            return self.df
        except Exception as e:
            print(f"❌ Error in create_age_groups: {e}")
            return self.df

    
    def create_bmi_categories(self):
        """Create WHO-standard BMI categories"""
        print("\n🏋️ CREATING BMI CATEGORIES")
        print("=" * 40)
        
        try:
            bins = [0, 18.5, 25, 30, 35, 40, 100]
            labels = ['Underweight', 'Normal', 'Overweight', 'Obese I', 'Obese II', 'Obese III']
            self.df['BMI_Category'] = pd.cut(self.df['BMI'], bins=bins, labels=labels)
            
            bmi_counts = self.df['BMI_Category'].value_counts()
            print("BMI Category Distribution:")
            for category, count in bmi_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  - {category}: {count} cases ({percentage:.1f}%)")
            
            self.transformation_report['bmi_categories'] = bmi_counts
            return self.df
        except KeyError:
            print("⚠️ Column 'BMI' not found. Skipping BMI category creation.")
            return self.df
        except Exception as e:
            print(f"❌ Error in create_bmi_categories: {e}")
            return self.df

    
    def create_glucose_categories(self):
        """Create clinical glucose categories"""
        print("\n🩸 CREATING GLUCOSE CATEGORIES")
        print("=" * 40)
        
        try:
            bins = [0, 99, 125, 200, 300]
            labels = ['Normal', 'Prediabetic', 'Diabetic', 'Severe']
            self.df['Glucose_Category'] = pd.cut(self.df['Glucose'], bins=bins, labels=labels)
            
            glucose_counts = self.df['Glucose_Category'].value_counts()
            for category, count in glucose_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  - {category}: {count} cases ({percentage:.1f}%)")
            
            self.transformation_report['glucose_categories'] = glucose_counts
            return self.df
        except KeyError:
            print("⚠️ Column 'Glucose' not found. Skipping glucose categories.")
            return self.df
        except Exception as e:
            print(f"❌ Error in create_glucose_categories: {e}")
            return self.df

    
    def create_blood_pressure_categories(self):
        """Create clinical blood pressure categories"""
        print("\n💓 CREATING BLOOD PRESSURE CATEGORIES")
        print("=" * 40)
        
        try:
            bins = [0, 60, 80, 90, 120]
            labels = ['Low', 'Normal', 'Elevated', 'High']
            self.df['BP_Category'] = pd.cut(self.df['BloodPressure'], bins=bins, labels=labels)
            
            bp_counts = self.df['BP_Category'].value_counts()
            for category, count in bp_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  - {category}: {count} cases ({percentage:.1f}%)")
            
            self.transformation_report['bp_categories'] = bp_counts
            return self.df
        except KeyError:
            print("⚠️ Column 'BloodPressure' not found. Skipping BP categories.")
            return self.df
        except Exception as e:
            print(f"❌ Error in create_blood_pressure_categories: {e}")
            return self.df

    def encode_categorical_features(self):
        """Encode categorical features using appropriate encoding methods"""
        print("\n🔤 ENCODING CATEGORICAL FEATURES")
        print("=" * 40)
        
        # Label encoding for ordinal features
        ordinal_features = ['BMI_Category', 'Glucose_Category', 'BP_Category']
        
        for feature in ordinal_features:
            if feature in self.df.columns:
                le = LabelEncoder()
                self.df[f'{feature}_Encoded'] = le.fit_transform(self.df[feature])
                print(f"  - {feature}: Label encoded")
        
        # One-hot encoding for nominal features
        nominal_features = ['Age_Group']
        
        for feature in nominal_features:
            if feature in self.df.columns:
                dummies = pd.get_dummies(self.df[feature], prefix=feature)
                self.df = pd.concat([self.df, dummies], axis=1)
                print(f"  - {feature}: One-hot encoded ({len(dummies.columns)} new features)")
        
        return self.df
    
    def scale_numerical_features(self, method='standard'):
        """Scale numerical features using specified method"""
        print(f"\n⚖️ SCALING NUMERICAL FEATURES USING {method.upper()} SCALING")
        print("=" * 40)
        
        try:
            numerical_features = ['Pregnancies', 'Glucose', 'BloodPressure', 
                                'SkinThickness', 'Insulin', 'BMI', 
                                'DiabetesPedigreeFunction', 'Age']
            
            available_features = [col for col in numerical_features if col in self.df.columns]
            if not available_features:
                print("⚠️ No numerical features found to scale.")
                return self.df
            
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Method must be 'standard' or 'minmax'")
            
            scaled_features = scaler.fit_transform(self.df[available_features])
            scaled_df = pd.DataFrame(scaled_features, columns=[f'{col}_Scaled' for col in available_features])
            
            self.df = pd.concat([self.df, scaled_df], axis=1)
            print(f"✅ Scaling applied successfully to {len(available_features)} features.")
            
            self.transformation_report['scaling'] = {'method': method, 'features': available_features}
            return self.df
        
        except ValueError as e:
            print(f"⚠️ Invalid scaling method: {e}")
            return self.df
        except Exception as e:
            print(f"❌ Error during scaling: {e}")
            return self.df

    
    def compare_scaling_methods(self):
        """Compare StandardScaler vs MinMaxScaler"""
        print("\n📊 COMPARING SCALING METHODS")
        print("=" * 40)
        
        numerical_features = ['Glucose', 'BMI', 'Age', 'Insulin']
        
        # Original data
        original_data = self.df[numerical_features]
        
        # Standard scaling
        standard_scaler = StandardScaler()
        standard_scaled = standard_scaler.fit_transform(original_data)
        
        # MinMax scaling
        minmax_scaler = MinMaxScaler()
        minmax_scaled = minmax_scaler.fit_transform(original_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, feature in enumerate(numerical_features):
            row, col = i // 2, i % 2
            
            axes[row, col].hist(original_data[feature], alpha=0.7, label='Original', bins=20)
            axes[row, col].hist(standard_scaled[:, i], alpha=0.7, label='Standard Scaled', bins=20)
            axes[row, col].hist(minmax_scaled[:, i], alpha=0.7, label='MinMax Scaled', bins=20)
            
            axes[row, col].set_title(f'{feature} - Scaling Comparison')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Statistical comparison
        print("\nScaling Method Comparison:")
        print("StandardScaler - Mean ~0, Std ~1")
        print("MinMaxScaler - Range [0, 1]")
        
        return {
            'standard_scaler': standard_scaler,
            'minmax_scaler': minmax_scaler
        }
    
    def create_interaction_features(self):
        """Create clinically meaningful interaction features"""
        print("\n🔗 CREATING INTERACTION FEATURES")
        print("=" * 40)
        
        # BMI-Age interaction (metabolic health)
        self.df['BMI_Age_Interaction'] = self.df['BMI'] * self.df['Age']
        
        # Glucose-Insulin interaction (insulin resistance proxy)
        self.df['Glucose_Insulin_Interaction'] = self.df['Glucose'] * (self.df['Insulin'] / 100)
        
        # Metabolic syndrome score (simplified)
        self.df['Metabolic_Score'] = (
            self.df['Glucose'] / 100 + 
            self.df['BMI'] / 30 + 
            self.df['BloodPressure'] / 80
        )
        
        print("Interaction Features Created:")
        print("  - BMI_Age_Interaction: Metabolic health indicator")
        print("  - Glucose_Insulin_Interaction: Insulin resistance proxy")
        print("  - Metabolic_Score: Combined metabolic health score")
        
        return self.df
    
    def generate_transformation_summary(self):
        """Generate comprehensive transformation summary"""
        print("\n📋 DATA TRANSFORMATION SUMMARY REPORT")
        print("=" * 60)
        
        print("🚀 TRANSFORMATION OPERATIONS PERFORMED:")
        if 'age_groups' in self.transformation_report:
            print("• Created clinically meaningful age groups")
        
        if 'bmi_categories' in self.transformation_report:
            print("• Created WHO-standard BMI categories")
        
        if 'glucose_categories' in self.transformation_report:
            print("• Created clinical glucose categories")
        
        if 'scaling' in self.transformation_report:
            print(f"• Applied {self.transformation_report['scaling']['method']} scaling to numerical features")
        
        print("\n📊 TRANSFORMED DATA STATUS:")
        print(f"• Total features: {self.df.shape[1]}")
        print(f"• Dataset shape: {self.df.shape}")
        
        # Count feature types
        numerical_count = len([col for col in self.df.columns if 'Scaled' in col or self.df[col].dtype in ['int64', 'float64']])
        categorical_count = len([col for col in self.df.columns if 'Category' in col or 'Group' in col])
        
        print(f"• Numerical features: {numerical_count}")
        print(f"• Categorical features: {categorical_count}")
        
        print("\n💡 RECOMMENDATIONS FOR PHASE 4:")
        print("• Proceed with feature selection and dimensionality reduction")
        print("• Analyze feature importance for model training")
        print("• Consider feature correlations for multicollinearity")
        
        return self.transformation_report

# Usage function

def execute_data_transformation_pipeline(df):
    """Execute complete data transformation pipeline"""
    transformer = DiabetesDataTransformer(df)
    
    try:
        transformer.create_age_groups()
        transformer.create_bmi_categories()
        transformer.create_glucose_categories()
        transformer.create_blood_pressure_categories()
        transformer.encode_categorical_features()
        transformer.compare_scaling_methods()
        transformer.scale_numerical_features(method='standard')
        transformer.create_interaction_features()
        transformation_report = transformer.generate_transformation_summary()
        return transformer.df, transformation_report
    
    except Exception as e:
        print(f"❌ Pipeline failed due to unexpected error: {e}")
        return df, {}
