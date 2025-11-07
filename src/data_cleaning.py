# File: src/data_cleaning.py
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class DiabetesDataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaning_report = {}
        
        # Define biological features where zero is impossible
        self.biological_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # Medical reasonable ranges (based on clinical knowledge)
        self.medical_ranges = {
            'Glucose': (50, 300),        # mg/dL (fasting glucose typically 70-125)
            'BloodPressure': (40, 120),   # mm Hg (diastolic)
            'SkinThickness': (10, 60),    # mm (triceps)
            'Insulin': (15, 200),         # mu U/ml (2-hour serum)
            'BMI': (15, 50),              # kg/m²
            'Age': (21, 81),              # From dataset min-max
            'Pregnancies': (0, 17)        # From dataset
        }
    
    def replace_zeros_with_nan(self):
        """Replace impossible zeros with NaN for proper imputation"""
        print("🔄 REPLACING IMPOSSIBLE ZEROS WITH NaN")
        print("=" * 50)
        
        zero_counts_before = {}
        zero_counts_after = {}
        
        for feature in self.biological_features:
            try:
                zero_count_before = (self.df[feature] == 0).sum()
                zero_counts_before[feature] = zero_count_before
                
                self.df[feature] = self.df[feature].replace(0, np.nan)
                
                zero_count_after = (self.df[feature] == 0).sum()
                zero_counts_after[feature] = zero_count_after
                
                print(f"{feature}: {zero_count_before} zeros → {zero_count_after} zeros")
            
            except KeyError:
                print(f"⚠️ Warning: Column '{feature}' not found in dataset.")
            except Exception as e:
                print(f"❌ Unexpected error in {feature}: {e}")
        
        self.cleaning_report['zero_replacement'] = {
            'before': zero_counts_before,
            'after': zero_counts_after
        }
        
        return self.df

    
    def analyze_missing_patterns(self):
        """Analyze missing data patterns after zero replacement"""
        print("\n📊 MISSING DATA ANALYSIS AFTER ZERO REPLACEMENT")
        print("=" * 50)
        
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'missing_count': missing_data,
            'missing_percentage': missing_percentage
        }).sort_values('missing_percentage', ascending=False)
        
        # Only show features with missing data
        missing_df = missing_df[missing_df['missing_count'] > 0]
        
        print(missing_df)
        
        # Visualize missing data
        plt.figure(figsize=(10, 6))
        missing_features = missing_df[missing_df['missing_percentage'] > 0].index
        missing_values = missing_df.loc[missing_features, 'missing_percentage']
        
        plt.barh(missing_features, missing_values)
        plt.xlabel('Percentage Missing (%)')
        plt.title('Missing Data After Zero Replacement')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        self.cleaning_report['missing_analysis'] = missing_df
        return missing_df
    
    def medical_range_validation(self):
        """Validate data against medical reasonable ranges"""
        print("\n🏥 MEDICAL RANGE VALIDATION")
        print("=" * 50)
        
        range_violations = {}
        
        for feature, (min_val, max_val) in self.medical_ranges.items():
            try:
                below_min = (self.df[feature] < min_val).sum()
                above_max = (self.df[feature] > max_val).sum()
                total_violations = below_min + above_max
                
                if total_violations > 0:
                    range_violations[feature] = {
                        'below_min': below_min,
                        'above_max': above_max,
                        'total_violations': total_violations,
                        'percentage': (total_violations / len(self.df)) * 100
                    }
                    
                    print(f"{feature}:")
                    print(f"  - Below {min_val}: {below_min} values")
                    print(f"  - Above {max_val}: {above_max} values")
                    print(f"  - Total violations: {total_violations} ({range_violations[feature]['percentage']:.1f}%)")
            except KeyError:
                print(f"⚠️ Warning: Missing column '{feature}' during range validation.")
            except Exception as e:
                print(f"❌ Unexpected error validating {feature}: {e}")
        
        if not range_violations:
            print("✅ All values within medically reasonable ranges")
        
        self.cleaning_report['range_validation'] = range_violations
        return range_violations

    
    def impute_missing_values(self, strategy='median'):
        """Impute missing values using specified strategy"""
        print(f"\n🔧 IMPUTING MISSING VALUES USING {strategy.upper()} STRATEGY")
        print("=" * 50)
        try:
        
        # Create imputer
            if strategy == 'median':
                imputer = SimpleImputer(strategy='median')
            elif strategy == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif strategy == 'most_frequent':
                imputer = SimpleImputer(strategy='most_frequent')
            else:
                raise ValueError("Strategy must be 'median', 'mean', or 'most_frequent'")
        
        # Impute biological features
            features_to_impute = [col for col in self.biological_features if self.df[col].isnull().sum() > 0]
            
            if features_to_impute:
                print(f"Imputing features: {features_to_impute}")
                
                # Store pre-imputation stats
                pre_impute_stats = self.df[features_to_impute].describe()
                
                # Perform imputation
                self.df[features_to_impute] = imputer.fit_transform(self.df[features_to_impute])
                
                # Store post-imputation stats
                post_impute_stats = self.df[features_to_impute].describe()
                
                print("\nImputation Summary:")
                for feature in features_to_impute:
                    imputed_count = self.df[feature].isnull().sum()
                    impute_value = imputer.statistics_[features_to_impute.index(feature)]
                    print(f"  - {feature}: {imputed_count} missing values imputed with {impute_value:.2f}")
                
                self.cleaning_report['imputation'] = {
                    'strategy': strategy,
                    'features': features_to_impute,
                    'pre_stats': pre_impute_stats,
                    'post_stats': post_impute_stats,
                    'imputer_values': imputer.statistics_
                }
            else:
                print("✅ No missing values to impute")
        except KeyError as e:
            print(f"⚠️ Missing column during imputation: {e}")
        except ValueError as e:
            print(f"⚠️ Invalid imputation strategy: {e}")
        except Exception as e:
            print(f"❌ Unexpected error during imputation: {e}")
            
            return self.df
    
    def detect_outliers_iqr(self, feature):
        """Detect outliers using IQR method for a single feature"""
        try:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[feature] < lower_bound) | (self.df[feature] > upper_bound)]
            
            return {
                'feature': feature,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(self.df)) * 100,
                'outliers': outliers[feature].values
            }
        except KeyError:
            print(f"⚠️ Missing column '{feature}' for outlier detection.")
            return {'feature': feature, 'outlier_count': 0, 'outlier_percentage': 0}
        except Exception as e:
            print(f"❌ Error detecting outliers in {feature}: {e}")
            return {'feature': feature, 'outlier_count': 0, 'outlier_percentage': 0}

    
    def comprehensive_outlier_analysis(self):
        """Perform comprehensive outlier analysis on all features"""
        print("\n📊 OUTLIER DETECTION ANALYSIS (IQR METHOD)")
        print("=" * 50)
        
        outlier_report = {}
        
        for feature in self.biological_features + ['Pregnancies', 'Age']:
            outlier_info = self.detect_outliers_iqr(feature)
            outlier_report[feature] = outlier_info
            
            if outlier_info['outlier_count'] > 0:
                print(f"{feature}:")
                print(f"  - Bounds: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]")
                print(f"  - Outliers: {outlier_info['outlier_count']} ({outlier_info['outlier_percentage']:.1f}%)")
        
        self.cleaning_report['outlier_analysis'] = outlier_report
        return outlier_report
    
    def generate_cleaning_summary(self):
        """Generate comprehensive cleaning summary report"""
        print("\n📋 DATA CLEANING SUMMARY REPORT")
        print("=" * 60)
        
        print("🚀 CLEANING OPERATIONS PERFORMED:")
        if 'zero_replacement' in self.cleaning_report:
            print("• Replaced biologically impossible zeros with NaN")
        
        if 'imputation' in self.cleaning_report:
            print(f"• Applied {self.cleaning_report['imputation']['strategy']} imputation")
        
        print("\n📊 CURRENT DATA STATUS:")
        print(f"• Total missing values: {self.df.isnull().sum().sum()}")
        print(f"• Dataset shape: {self.df.shape}")
        
        print("\n💡 RECOMMENDATIONS FOR PHASE 3:")
        print("• Proceed with feature engineering")
        print("• Consider outlier treatment based on domain knowledge")
        print("• Validate data quality after cleaning")
        
        return self.cleaning_report

# Usage function
def execute_data_cleaning_pipeline(df):
    """Execute complete data cleaning pipeline"""
    cleaner = DiabetesDataCleaner(df)
    
    try:
        cleaner.replace_zeros_with_nan()
        cleaner.analyze_missing_patterns()
        cleaner.medical_range_validation()
        cleaner.impute_missing_values(strategy='median')
        cleaner.comprehensive_outlier_analysis()
        cleaning_report = cleaner.generate_cleaning_summary()
        return cleaner.df, cleaning_report
    
    except Exception as e:
        print(f"❌ Pipeline failed due to unexpected error: {e}")
        return df, {}
