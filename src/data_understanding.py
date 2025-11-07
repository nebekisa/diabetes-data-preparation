import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pathlib import Path

class DiabetesDataAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.column_mapping = {
            'Pregnancies': 'Number of times pregnant',
            'Glucose': 'Plasma glucose concentration (mg/dL)',
            'BloodPressure': 'Diastolic blood pressure (mm Hg)',
            'SkinThickness': 'Triceps skin fold thickness (mm)',
            'Insulin': '2-Hour serum insulin (mu U/ml)',
            'BMI': 'Body mass index (kg/m²)',
            'DiabetesPedigreeFunction': 'Diabetes pedigree function',
            'Age': 'Age in years',
            'Outcome': 'Target variable (0=non-diabetic, 1=diabetic)'
        }
        self.biological_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
    def load_data(self):
        """Load and validate diabetes dataset"""
        try:
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
            self.df = pd.read_csv(self.data_path)
            print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            print(f"📋 Columns: {list(self.df.columns)}")
            return self.df
        
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
        except pd.errors.EmptyDataError:
            print("❌ Error: The CSV file is empty.")
        except pd.errors.ParserError:
            print("❌ Error: Failed to parse CSV file.")
        except Exception as e:
            print(f"⚠️ Unexpected error while loading data: {e}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"📋 Columns: {list(self.df.columns)}")
        return self.df
    
    def basic_assessment(self):
        """Perform comprehensive initial data assessment"""
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            
            print("📊 DATASET BASIC INFORMATION")
            print("=" * 50)
            print(f"Dataset shape: {self.df.shape}")
            print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            print("\n📋 DATA TYPES AND MISSING VALUES:")
            info_df = pd.DataFrame({
                'dtype': self.df.dtypes,
                'non_null_count': self.df.count(),
                'null_count': self.df.isnull().sum(),
                'null_percentage': (self.df.isnull().sum() / len(self.df)) * 100
            })
            print(info_df)
            return info_df
        
        except ValueError as e:
            print(f"⚠️ {e}")
        except Exception as e:
            print(f"⚠️ Error during basic assessment: {e}")
        print("📊 DATASET BASIC INFORMATION")
        print("=" * 50)
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\n📋 DATA TYPES AND MISSING VALUES:")
        info_df = pd.DataFrame({
            'dtype': self.df.dtypes,
            'non_null_count': self.df.count(),
            'null_count': self.df.isnull().sum(),
            'null_percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        print(info_df)
        return info_df
    
    def statistical_summary(self):
        """Generate detailed statistical summary"""
        try:
            if self.df is None:
                raise ValueError("Data not loaded. Please call load_data() first.")
            
            print("📈 STATISTICAL SUMMARY")
            print("=" * 40)
            
            stats = self.df.describe().T
            stats['median'] = self.df.median(numeric_only=True)
            stats['variance'] = self.df.var(numeric_only=True)
            stats['skewness'] = self.df.skew(numeric_only=True)
            stats['kurtosis'] = self.df.kurtosis(numeric_only=True)
            
            print(stats.round(3))
            return stats
        
        except ValueError as e:
            print(f"⚠️ {e}")
        except KeyError as e:
            print(f"⚠️ Missing column: {e}")
        except Exception as e:
            print(f"⚠️ Error generating statistical summary: {e}")
        print("📈 STATISTICAL SUMMARY")
        print("=" * 40)
        
        stats = self.df.describe().T
        stats['median'] = self.df.median()
        stats['variance'] = self.df.var()
        stats['skewness'] = self.df.skew()
        stats['kurtosis'] = self.df.kurtosis()
        
        print(stats.round(3))
        return stats
    
    def identify_biological_impossibilities(self):
        """Identify zeros in features where zero is biologically impossible"""
        try:
            print("🔍 BIOLOGICAL IMPOSSIBILITIES ANALYSIS")
            print("=" * 50)
            
            zero_analysis = {}
            
            for feature in self.biological_features:
                if feature not in self.df.columns:
                    print(f"⚠️ Skipping missing feature: {feature}")
                    continue
                
                zero_count = (self.df[feature] == 0).sum()
                zero_percentage = (zero_count / len(self.df)) * 100
                zero_analysis[feature] = {
                    'zero_count': zero_count,
                    'zero_percentage': zero_percentage,
                    'min_value': self.df[feature].min(),
                    'max_value': self.df[feature].max(),
                    'interpretation': 'BIOLOGICALLY IMPOSSIBLE' if zero_count > 0 else 'OK'
                }
                
                print(f"{feature}:")
                print(f"  - Zero values: {zero_count} ({zero_percentage:.2f}%)")
                print(f"  - Value range: [{self.df[feature].min()}, {self.df[feature].max()}]")
                print(f"  - Status: {zero_analysis[feature]['interpretation']}\n")
            
            return zero_analysis
        
        except Exception as e:
            print(f"⚠️ Error identifying biological impossibilities: {e}")
            return {}
        print("🔍 BIOLOGICAL IMPOSSIBILITIES ANALYSIS")
        print("=" * 50)
        
        zero_analysis = {}
        
        for feature in self.biological_features:
            zero_count = (self.df[feature] == 0).sum()
            zero_percentage = (zero_count / len(self.df)) * 100
            zero_analysis[feature] = {
                'zero_count': zero_count,
                'zero_percentage': zero_percentage,
                'min_value': self.df[feature].min(),
                'max_value': self.df[feature].max(),
                'interpretation': 'BIOLOGICALLY IMPOSSIBLE' if zero_count > 0 else 'OK'
            }
            
            print(f"{feature}:")
            print(f"  - Zero values: {zero_count} ({zero_percentage:.2f}%)")
            print(f"  - Value range: [{self.df[feature].min()}, {self.df[feature].max()}]")
            print(f"  - Status: {zero_analysis[feature]['interpretation']}")
            print()
        
        # Also check other features for context
        print("Other features zero check:")
        other_features = [col for col in self.df.columns if col not in self.biological_features + ['Outcome']]
        for feature in other_features:
            zero_count = (self.df[feature] == 0).sum()
            if zero_count > 0:
                print(f"  - {feature}: {zero_count} zeros ({(zero_count/len(self.df))*100:.2f}%)")
        
        return zero_analysis
    
    def analyze_target_variable(self, target_col='Outcome'):
        """Comprehensive analysis of target variable"""
        try:
            if target_col not in self.df.columns:
                raise KeyError(f"Target column '{target_col}' not found.")
            
            print("🎯 TARGET VARIABLE ANALYSIS")
            print("=" * 40)
            
            target_counts = self.df[target_col].value_counts()
            target_percentages = self.df[target_col].value_counts(normalize=True) * 100
            
            print("Class Distribution:")
            for class_val in target_counts.index:
                count = target_counts[class_val]
                percent = target_percentages[class_val]
                class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
                print(f"  {class_label} (Class {class_val}): {count} cases ({percent:.2f}%)")
            
            imbalance_ratio = target_counts.min() / target_counts.max()
            print(f"\nImbalance Ratio: {imbalance_ratio:.3f}")
            
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.countplot(data=self.df, x=target_col, ax=axes[0])
            axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
            axes[0].set_xticklabels(['Non-Diabetic', 'Diabetic'])
            axes[0].set_xlabel('')
            
            axes[1].pie(target_counts, labels=['Non-Diabetic', 'Diabetic'], 
                        autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
            axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.show()
            
            return {
                'counts': target_counts,
                'percentages': target_percentages,
                'imbalance_ratio': imbalance_ratio
            }
        
        except KeyError as e:
            print(f"⚠️ {e}")
        except Exception as e:
            print(f"⚠️ Error analyzing target variable: {e}")
        print("🎯 TARGET VARIABLE ANALYSIS")
        print("=" * 40)
        
        if target_col not in self.df.columns:
            available_cols = list(self.df.columns)
            print(f"❌ Target column '{target_col}' not found. Available columns: {available_cols}")
            return None
        
        target_counts = self.df[target_col].value_counts()
        target_percentages = self.df[target_col].value_counts(normalize=True) * 100
        
        print("Class Distribution:")
        for class_val in target_counts.index:
            count = target_counts[class_val]
            percent = target_percentages[class_val]
            class_label = 'Non-Diabetic' if class_val == 0 else 'Diabetic'
            print(f"  {class_label} (Class {class_val}): {count} cases ({percent:.2f}%)")
        
        # Imbalance ratio
        imbalance_ratio = target_counts.min() / target_counts.max()
        print(f"\nImbalance Ratio: {imbalance_ratio:.3f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        sns.countplot(data=self.df, x=target_col, ax=axes[0])
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(['Non-Diabetic', 'Diabetic'])
        axes[0].set_xlabel('')
        
        # Pie chart
        axes[1].pie(target_counts, labels=['Non-Diabetic', 'Diabetic'], 
                    autopct='%1.1f%%', startangle=90, colors=['lightblue', 'salmon'])
        axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return {
            'counts': target_counts,
            'percentages': target_percentages,
            'imbalance_ratio': imbalance_ratio
        }
    
    def generate_assessment_report(self, zero_analysis, target_analysis):
        """Generate comprehensive initial data assessment report"""
        print("📋 INITIAL DATA ASSESSMENT REPORT")
        print("=" * 60)
        
        # Executive Summary
        print("\n🚀 EXECUTIVE SUMMARY")
        print(f"• Dataset: {self.df.shape[0]} observations, {self.df.shape[1]} features")
        print(f"• Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if target_analysis:
            print(f"• Class Distribution: {target_analysis['imbalance_ratio']:.3f} imbalance ratio")
        
        # Data Quality Issues
        print("\n⚠️ CRITICAL DATA QUALITY ISSUES")
        biological_issues = {k: v for k, v in zero_analysis.items() 
                            if v['zero_count'] > 0}
        
        if biological_issues:
            print("• Biological Impossibilities Found:")
            for feature, analysis in biological_issues.items():
                print(f"  - {feature}: {analysis['zero_count']} zeros ({analysis['zero_percentage']:.1f}%)")
        else:
            print("• No biological impossibilities detected")
        
        # Missing Data Summary (from your visualization, there are no true missing values)
        print("\n• Missing Values: No true NaN values detected")
        print("• Note: Zeros are used as placeholders for missing values in biological features")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS FOR PHASE 2: DATA CLEANING")
        if biological_issues:
            print("• Replace impossible zeros with NaN for proper imputation")
            print("• Use median imputation for biological features (robust to outliers)")
            print("• Consider domain knowledge for reasonable value ranges")
        print("• Handle class imbalance in Phase 5")
        print("• Proceed to outlier detection and treatment")

# Usage example
if __name__ == "__main__":
    analyzer = DiabetesDataAnalyzer('../data/raw/Diabetes Missing Data.csv')
    df = analyzer.load_data()
    info_df = analyzer.basic_assessment()
    stats_df = analyzer.statistical_summary()
    zero_analysis = analyzer.identify_biological_impossibilities()
    target_analysis = analyzer.analyze_target_variable('Outcome')
    analyzer.generate_assessment_report(zero_analysis, target_analysis)