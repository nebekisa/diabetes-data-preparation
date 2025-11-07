# File: src/data_reduction.py (UPDATED VERSION)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataReducer:
    def __init__(self, df, target_col='Outcome'):
        self.df = df.copy()
        self.target_col = target_col
        self.reduction_report = {}
        
        # Separate features and target
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        
        # Prepare numerical features only for correlation analysis
        self.numerical_features = self.X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"📊 Feature Types Identified:")
        print(f"  - Numerical features: {len(self.numerical_features)}")
        print(f"  - Categorical features: {len(self.categorical_features)}")
    
    def get_feature_categories(self):
        """Categorize features for better analysis"""
        feature_categories = {
            'original_numerical': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
            'scaled_features': [col for col in self.X.columns if '_Scaled' in col],
            'clinical_categories': [col for col in self.X.columns if 'Category' in col or 'Group' in col],
            'interaction_features': ['BMI_Age_Interaction', 'Glucose_Insulin_Interaction', 'Metabolic_Score'],
            'encoded_features': [col for col in self.X.columns if 'Encoded' in col],
            'one_hot_features': [col for col in self.X.columns if 'Age_Group' in col and 'Encoded' not in col]
        }
        return feature_categories
    
    def analyze_feature_correlations(self):
        """Comprehensive correlation analysis - NUMERICAL FEATURES ONLY"""
        print("🔗 COMPREHENSIVE FEATURE CORRELATION ANALYSIS (Numerical Features Only)")
        print("=" * 50)
        
        # Use only numerical features for correlation
        X_numerical = self.X[self.numerical_features]
        corr_matrix = X_numerical.corr()
        
        # Find highly correlated features (absolute correlation > 0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i], 
                        corr_matrix.columns[j], 
                        corr_matrix.iloc[i, j]
                    ))
        
        print("🚨 HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.8):")
        if high_corr_pairs:
            for feat1, feat2, corr in high_corr_pairs:
                print(f"  - {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print("  - No highly correlated feature pairs found")
        
        # Correlation with target (numerical features only)
        target_correlations = {}
        for col in X_numerical.columns:
            corr = np.corrcoef(X_numerical[col], self.y)[0, 1]
            target_correlations[col] = corr
        
        # Sort by absolute correlation with target
        sorted_correlations = sorted(target_correlations.items(), 
                                   key=lambda x: abs(x[1]), reverse=True)
        
        print("\n📊 TOP NUMERICAL FEATURES BY CORRELATION WITH TARGET:")
        for feature, corr in sorted_correlations[:10]:
            print(f"  - {feature}: {corr:.3f}")
        
        # Visualization - Numerical features only
        plt.figure(figsize=(12, 10))
        
        # Top correlations heatmap (numerical features only)
        top_features = [feat for feat, _ in sorted_correlations[:15]] 
        if self.target_col in self.df.columns:
            # Include target in correlation matrix if available
            top_corr_matrix = self.df[top_features + [self.target_col]].corr()
        else:
            top_corr_matrix = self.df[top_features].corr()
        
        sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, fmt='.2f')
        plt.title('Top Numerical Features Correlation with Diabetes Outcome', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        self.reduction_report['correlation_analysis'] = {
            'high_corr_pairs': high_corr_pairs,
            'target_correlations': dict(sorted_correlations),
            'correlation_matrix': corr_matrix
        }
        
        return high_corr_pairs, dict(sorted_correlations)
    
    def prepare_features_for_analysis(self):
        """Prepare all features for feature selection (handle categorical)"""
        print("\n🔄 PREPARING FEATURES FOR ANALYSIS")
        print("=" * 50)
        
        # Start with numerical features
        X_processed = self.X[self.numerical_features].copy()
        
        # Handle categorical features by encoding them
        if self.categorical_features:
            print("Encoding categorical features for analysis:")
            for cat_feature in self.categorical_features:
                if cat_feature in self.X.columns:
                    # Use label encoding for categorical features
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(self.X[cat_feature].astype(str))
                    X_processed[f"{cat_feature}_Encoded"] = encoded_values
                    print(f"  - {cat_feature} → {cat_feature}_Encoded")
        
        print(f"Final feature set for analysis: {X_processed.shape[1]} features")
        return X_processed
    
    def mutual_information_analysis(self):
        """Feature selection using Mutual Information"""
        print("\n🎯 MUTUAL INFORMATION FEATURE SELECTION")
        print("=" * 50)
        
        # Prepare all features (numerical + encoded categorical)
        X_processed = self.prepare_features_for_analysis()
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X_processed, self.y, random_state=42)
        mi_features = pd.DataFrame({
            'feature': X_processed.columns,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)
        
        print("TOP FEATURES BY MUTUAL INFORMATION:")
        for _, row in mi_features.head(10).iterrows():
            print(f"  - {row['feature']}: {row['mi_score']:.4f}")
        
        # Visualization
        plt.figure(figsize=(10, 8))
        top_mi = mi_features.head(15)
        plt.barh(top_mi['feature'], top_mi['mi_score'])
        plt.xlabel('Mutual Information Score')
        plt.title('Top Features by Mutual Information with Diabetes Outcome')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        self.reduction_report['mutual_information'] = mi_features
        return mi_features
    
    def select_kbest_analysis(self, k=10):
        """SelectKBest feature selection analysis"""
        print(f"\n🏆 SELECTKBEST FEATURE SELECTION (Top {k} Features)")
        print("=" * 50)
        
        X_processed = self.prepare_features_for_analysis()
        
        # Use f_classif for classification
        selector = SelectKBest(score_func=f_classif, k=min(k, X_processed.shape[1]))
        selector.fit(X_processed, self.y)
        
        kbest_scores = pd.DataFrame({
            'feature': X_processed.columns,
            'f_score': selector.scores_,
            'p_value': selector.pvalues_
        }).sort_values('f_score', ascending=False)
        
        print("TOP FEATURES BY F-SCORE (ANOVA):")
        for _, row in kbest_scores.head(k).iterrows():
            print(f"  - {row['feature']}: F={row['f_score']:.1f}, p={row['p_value']:.4f}")
        
        self.reduction_report['selectkbest'] = kbest_scores
        return kbest_scores
    
    def random_forest_feature_importance(self):
        """Feature importance using Random Forest"""
        print("\n🌲 RANDOM FOREST FEATURE IMPORTANCE")
        print("=" * 50)
        
        X_processed = self.prepare_features_for_analysis()
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_processed, self.y)
        
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("TOP FEATURES BY RANDOM FOREST IMPORTANCE:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  - {row['feature']}: {row['importance']:.4f}")
        
        # Visualization
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance for Diabetes Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        self.reduction_report['random_forest'] = feature_importance
        return feature_importance
    
    def recursive_feature_elimination(self, n_features=10):
        """Recursive Feature Elimination"""
        print(f"\n🔄 RECURSIVE FEATURE ELIMINATION (Top {n_features} Features)")
        print("=" * 50)
        
        X_processed = self.prepare_features_for_analysis()
        
        # Use logistic regression as estimator
        estimator = LogisticRegression(random_state=42, max_iter=1000)
        n_features_to_select = min(n_features, X_processed.shape[1])
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector.fit(X_processed, self.y)
        
        rfe_ranking = pd.DataFrame({
            'feature': X_processed.columns,
            'rfe_ranking': selector.ranking_,
            'selected': selector.support_
        }).sort_values('rfe_ranking')
        
        print("RFE SELECTED FEATURES:")
        selected_features = rfe_ranking[rfe_ranking['selected'] == True]
        for _, row in selected_features.iterrows():
            print(f"  - {row['feature']} (Rank: {row['rfe_ranking']})")
        
        self.reduction_report['rfe'] = rfe_ranking
        return rfe_ranking
    
    def pca_analysis(self):
        """Principal Component Analysis for dimensionality reduction - NUMERICAL FEATURES ONLY"""
        print("\n🔍 PRINCIPAL COMPONENT ANALYSIS (PCA) - Numerical Features Only")
        print("=" * 50)
        
        # Use only scaled numerical features for PCA
        scaled_features = [col for col in self.numerical_features if '_Scaled' in col]
        if not scaled_features:
            # If no scaled features, use original numerical features
            scaled_features = [col for col in self.numerical_features if col not in ['Outcome', 'Class']]
        
        X_scaled = self.X[scaled_features]
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("PCA EXPLAINED VARIANCE:")
        for i, (var, cum_var) in enumerate(zip(explained_variance, cumulative_variance)):
            print(f"  - PC{i+1}: {var:.3f} ({cum_var:.3f} cumulative)")
        
        # Determine optimal number of components (95% variance)
        optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
        print(f"\n💡 OPTIMAL COMPONENTS: {optimal_components} (95% variance explained)")
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scree plot
        ax1.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6)
        ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'r-', marker='o')
        ax1.set_xlabel('Principal Components')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('PCA Scree Plot')
        ax1.legend(['Cumulative', 'Individual'])
        ax1.grid(True, alpha=0.3)
        
        # PCA loadings (first two components)
        loadings = pd.DataFrame(
            pca.components_[:2].T,
            columns=['PC1', 'PC2'],
            index=scaled_features
        )
        
        # Plot loadings
        ax2.scatter(loadings['PC1'], loadings['PC2'], s=100)
        for i, feature in enumerate(scaled_features):
            ax2.annotate(feature.replace('_Scaled', ''), 
                        (loadings.iloc[i, 0], loadings.iloc[i, 1]),
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.axhline(y=0, color='grey', linestyle='--', alpha=0.7)
        ax2.axvline(x=0, color='grey', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.set_title('PCA Loadings (First Two Components)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        self.reduction_report['pca'] = {
            'explained_variance': explained_variance,
            'cumulative_variance': cumulative_variance,
            'optimal_components': optimal_components,
            'loadings': loadings,
            'pca_model': pca
        }
        
        return pca, explained_variance, optimal_components
    
    def create_feature_selection_summary(self):
        """Create comprehensive feature selection summary"""
        print("\n📊 FEATURE SELECTION CONSENSUS SUMMARY")
        print("=" * 50)
        
        # Get top features from each method
        mi_top = set(self.reduction_report['mutual_information'].head(10)['feature'])
        rf_top = set(self.reduction_report['random_forest'].head(10)['feature'])
        kbest_top = set(self.reduction_report['selectkbest'].head(10)['feature'])
        rfe_top = set(self.reduction_report['rfe'][self.reduction_report['rfe']['selected'] == True]['feature'])
        
        # Find consensus features
        all_methods = [mi_top, rf_top, kbest_top, rfe_top]
        consensus_features = set.intersection(*all_methods)
        
        print("🎯 CONSENSUS TOP FEATURES (Selected by ALL methods):")
        if consensus_features:
            for feature in consensus_features:
                print(f"  - {feature}")
        else:
            print("  - No features selected by all methods")
            # Find features selected by 3 out of 4 methods
            feature_votes = {}
            for feature_set in all_methods:
                for feature in feature_set:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1
            
            high_agreement = {feat: votes for feat, votes in feature_votes.items() if votes >= 3}
            if high_agreement:
                print("\n🏅 HIGH AGREEMENT FEATURES (3+ methods):")
                for feat, votes in sorted(high_agreement.items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {feat} ({votes} methods)")
        
        return consensus_features if consensus_features else high_agreement
    
    def apply_final_feature_selection(self, n_features=15):
        """Apply final feature selection based on comprehensive analysis"""
        print(f"\n🎯 APPLYING FINAL FEATURE SELECTION (Top {n_features} Features)")
        print("=" * 50)
        
        # Combine scores from different methods
        feature_scores = {}
        
        # Normalize and combine scores
        methods = ['mutual_information', 'random_forest', 'selectkbest']
        for method in methods:
            df = self.reduction_report[method]
            # Get the score column name
            score_column = ['mi_score', 'importance', 'f_score'][methods.index(method)]
            max_score = df[score_column].max()
            
            for _, row in df.iterrows():
                feature = row['feature']
                score = row[score_column]
                normalized_score = score / max_score
                
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(normalized_score)
        
        # Calculate average normalized score
        avg_scores = {feat: np.mean(scores) for feat, scores in feature_scores.items()}
        
        # Select top features
        selected_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        
        print("FINAL SELECTED FEATURES:")
        final_features = []
        for feature, score in selected_features:
            print(f"  - {feature}: {score:.3f}")
            final_features.append(feature)
        
        # Create reduced dataset - use original feature names (not encoded versions)
        original_feature_names = []
        for feature in final_features:
            # Convert encoded names back to original if possible
            if feature.endswith('_Encoded'):
                original_name = feature.replace('_Encoded', '')
                if original_name in self.df.columns:
                    original_feature_names.append(original_name)
                else:
                    original_feature_names.append(feature)
            else:
                original_feature_names.append(feature)
        
        # Ensure we have unique features
        original_feature_names = list(set(original_feature_names))
        
        # Create final dataset with selected features
        reduced_df = self.df[original_feature_names + [self.target_col]]
        
        print(f"\n📉 DATASET REDUCTION: {self.df.shape[1]} → {reduced_df.shape[1]} features")
        print(f"📊 FEATURE REDUCTION: {len(self.X.columns)} → {len(original_feature_names)} predictors")
        
        self.reduction_report['final_selection'] = {
            'selected_features': original_feature_names,
            'feature_scores': avg_scores,
            'reduced_dataset': reduced_df
        }
        
        return reduced_df, original_feature_names
    
    def generate_reduction_summary(self):
        """Generate comprehensive reduction summary report"""
        print("\n📋 DATA REDUCTION SUMMARY REPORT")
        print("=" * 60)
        
        print("🚀 REDUCTION OPERATIONS PERFORMED:")
        print("• Comprehensive correlation analysis (numerical features)")
        print("• Mutual information feature selection")
        print("• SelectKBest (ANOVA) analysis")
        print("• Random Forest feature importance")
        print("• Recursive Feature Elimination")
        print("• Principal Component Analysis")
        print("• Consensus feature selection")
        
        print("\n📊 KEY FINDINGS:")
        if 'correlation_analysis' in self.reduction_report:
            high_corr = len(self.reduction_report['correlation_analysis']['high_corr_pairs'])
            print(f"• High correlation pairs: {high_corr}")
        
        if 'pca' in self.reduction_report:
            optimal_comp = self.reduction_report['pca']['optimal_components']
            print(f"• PCA optimal components: {optimal_comp}")
        
        if 'final_selection' in self.reduction_report:
            final_count = len(self.reduction_report['final_selection']['selected_features'])
            print(f"• Final selected features: {final_count}")
        
        print("\n💡 RECOMMENDATIONS FOR PHASE 5:")
        print("• Address class imbalance using selected features")
        print("• Validate feature selection with model performance")
        print("• Consider domain knowledge for final feature set")
        
        return self.reduction_report

# Usage function (unchanged)
def execute_data_reduction_pipeline(df, target_col='Outcome', n_final_features=15):
    """Execute complete data reduction pipeline"""
    reducer = DiabetesDataReducer(df, target_col)
    
    # Step 1: Correlation analysis
    reducer.analyze_feature_correlations()
    
    # Step 2: Multiple feature selection methods
    reducer.mutual_information_analysis()
    reducer.select_kbest_analysis()
    reducer.random_forest_feature_importance()
    reducer.recursive_feature_elimination()
    
    # Step 3: Dimensionality reduction
    reducer.pca_analysis()
    
    # Step 4: Consensus feature selection
    reducer.create_feature_selection_summary()
    
    # Step 5: Final feature selection
    reduced_df, selected_features = reducer.apply_final_feature_selection(n_final_features)
    
    # Step 6: Generate summary
    reduction_report = reducer.generate_reduction_summary()
    
    return reduced_df, selected_features, reduction_report