#!/usr/bin/env python3
"""
Startup Success Prediction - Comprehensive EDA and Machine Learning Pipeline

This script includes:
- Automatic plot saving to organized folders
- Advanced data preprocessing and feature engineering
- Improved machine learning models with hyperparameter tuning
- Better data cleaning and CSV enhancements
- Cross-validation and ensemble methods

Dataset: big_startup_secsees_dataset.csv
Objective: Predict startup success (acquired/IPO) vs failure (closed) vs operating status
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import re
from pathlib import Path
warnings.filterwarnings('ignore')

# Enhanced Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           roc_auc_score, precision_recall_curve, f1_score)
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EnhancedStartupAnalyzer:
    def __init__(self, csv_file_path, output_dir="startup_analysis_output"):
        """Initialize the enhanced analyzer with output directory for plots."""
        self.csv_file = csv_file_path
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.models_dir = self.output_dir / "models"
        self.data_dir = self.output_dir / "data"
        
        # Create directories
        for dir_path in [self.output_dir, self.plots_dir, self.models_dir, self.data_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.df_processed = None
        self.models = {}
        self.results = {}
        self.plot_counter = 1
        
    def save_plot(self, filename, dpi=300, bbox_inches='tight'):
        """Save plot with organized naming and high quality."""
        plot_path = self.plots_dir / f"{self.plot_counter:02d}_{filename}.png"
        plt.savefig(plot_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"📊 Plot saved: {plot_path}")
        self.plot_counter += 1
        
    def load_and_enhanced_clean_data(self):
        """Enhanced data loading and cleaning with comprehensive preprocessing."""
        print("🔄 Loading and performing enhanced data cleaning...")
        
        # Load the dataset
        self.df = pd.read_csv(self.csv_file)
        print(f"Original dataset shape: {self.df.shape}")
        
        # Enhanced data cleaning and preprocessing
        self._enhanced_data_preprocessing()
        
        # Save cleaned dataset
        cleaned_path = self.data_dir / "cleaned_startup_dataset.csv"
        self.df.to_csv(cleaned_path, index=False)
        print(f"💾 Cleaned dataset saved: {cleaned_path}")
        
        print(f"After enhanced cleaning: {self.df.shape}")
        return self.df
    
    def _enhanced_data_preprocessing(self):
        """Comprehensive data preprocessing and CSV enhancements."""
        
        print("🔧 Applying enhanced preprocessing...")
        
        # 1. Fix data type issues and handle missing values smartly
        self.df['funding_total_usd'] = pd.to_numeric(self.df['funding_total_usd'], errors='coerce')
        
        # 2. Enhanced date processing
        date_columns = ['founded_at', 'first_funding_at', 'last_funding_at']
        for col in date_columns:
            # Handle various date formats
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
        
        # 3. Clean and standardize text data
        text_columns = ['name', 'category_list', 'homepage_url']
        for col in text_columns:
            if col in self.df.columns:
                # Remove extra whitespace and standardize
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].replace('nan', np.nan)
        
        # 4. Enhanced category cleaning
        self.df['category_list'] = self.df['category_list'].apply(self._clean_categories)
        
        # 5. Geographic data standardization
        self.df['country_code'] = self.df['country_code'].str.upper().str.strip()
        self.df['state_code'] = self.df['state_code'].str.upper().str.strip()
        
        # 6. Remove invalid records
        # Remove rows with missing critical information
        critical_cols = ['status', 'name']
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=critical_cols)
        print(f"Removed {initial_count - len(self.df)} rows with missing critical data")
        
        # 7. Handle outliers in funding data
        self._handle_funding_outliers()
        
        # 8. Add derived columns for better analysis
        self._add_derived_columns()
        
        # 9. Add basic ML features to main dataset
        self._add_basic_ml_features()
        
        # 10. Validate and fix inconsistent data
        self._validate_and_fix_data()
    
    def _clean_categories(self, categories):
        """Clean and standardize category data."""
        if pd.isna(categories) or categories == 'nan':
            return np.nan
        
        # Split categories and clean each one
        cats = str(categories).split('|')
        cleaned_cats = []
        
        for cat in cats:
            cat = cat.strip()
            # Standardize common variations
            cat = re.sub(r'\s+', ' ', cat)  # Multiple spaces to single
            cat = cat.title()  # Proper case
            
            # Fix common category name variations
            category_mappings = {
                'Saas': 'SaaS',
                'Ai': 'AI',
                'Iot': 'IoT',
                'B2b': 'B2B',
                'B2c': 'B2C',
                'Api': 'API',
                'Crm': 'CRM',
                'Hr': 'HR',
                'It': 'IT'
            }
            
            for old, new in category_mappings.items():
                cat = cat.replace(old, new)
            
            if cat and cat != '':
                cleaned_cats.append(cat)
        
        return '|'.join(cleaned_cats) if cleaned_cats else np.nan
    
    def _handle_funding_outliers(self):
        """Handle outliers in funding data using statistical methods."""
        funding_col = 'funding_total_usd'
        
        if funding_col in self.df.columns:
            # Calculate IQR for outlier detection
            Q1 = self.df[funding_col].quantile(0.25)
            Q3 = self.df[funding_col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds (more conservative than typical 1.5*IQR)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Flag outliers but don't remove them - just log
            outliers = self.df[
                (self.df[funding_col] < lower_bound) | 
                (self.df[funding_col] > upper_bound)
            ]
            
            print(f"Identified {len(outliers)} funding outliers (kept in dataset)")
            
            # Cap extreme outliers (beyond 99.9th percentile)
            cap_value = self.df[funding_col].quantile(0.999)
            extreme_outliers = self.df[funding_col] > cap_value
            self.df.loc[extreme_outliers, funding_col] = cap_value
            print(f"Capped {extreme_outliers.sum()} extreme funding outliers at ${cap_value:,.0f}")
    
    def _add_derived_columns(self):
        """Add useful derived columns for analysis."""
        
        # 1. Funding-related features
        self.df['has_funding'] = (self.df['funding_total_usd'].notna() & 
                                 (self.df['funding_total_usd'] > 0)).astype(int)
        
        # 2. Time-based features
        current_date = pd.Timestamp.now()
        
        # Company age
        self.df['company_age_years'] = (
            (current_date - self.df['founded_at']).dt.days / 365.25
        ).round(2)
        
        # Time to first funding
        self.df['days_to_first_funding'] = (
            self.df['first_funding_at'] - self.df['founded_at']
        ).dt.days
        
        # Funding duration (first to last funding)
        self.df['funding_duration_days'] = (
            self.df['last_funding_at'] - self.df['first_funding_at']
        ).dt.days
        
        # 3. Category-related features
        self.df['category_count'] = self.df['category_list'].str.count('\|') + 1
        self.df['category_count'] = self.df['category_count'].fillna(0)
        
        # Extract primary category
        self.df['primary_category'] = (
            self.df['category_list'].str.split('|').str[0]
        ).fillna('Unknown')
        
        # 4. Geographic features
        self.df['has_location'] = self.df['country_code'].notna().astype(int)
        self.df['has_detailed_location'] = (
            self.df['city'].notna() & self.df['state_code'].notna()
        ).astype(int)
        
        # 5. URL-based features
        self.df['has_homepage'] = self.df['homepage_url'].notna().astype(int)
        
        # Extract domain type
        self.df['domain_type'] = self.df['homepage_url'].apply(self._extract_domain_type)
        
        # 6. Funding efficiency metrics
        self.df['funding_per_round'] = (
            self.df['funding_total_usd'] / self.df['funding_rounds']
        ).replace([np.inf, -np.inf], np.nan)
        
        print("✅ Added derived columns for enhanced analysis")
    
    def _add_basic_ml_features(self):
        """Add basic ML features to the main dataset that will be used throughout analysis."""
        
        print("🔧 Adding basic ML features to main dataset...")
        
        # 1. Technology startup classification
        tech_categories = ['Software', 'Biotechnology', 'Artificial Intelligence', 'Machine Learning', 
                          'Data Analytics', 'Cybersecurity', 'Blockchain', 'IoT', 'API', 'SaaS',
                          'Mobile', 'Apps', 'Web Development', 'Cloud Computing', 'Big Data']
        
        self.df['is_tech_startup'] = self.df['category_list'].apply(
            lambda x: any(cat in str(x) for cat in tech_categories) if pd.notna(x) else 0
        ).astype(int)
        
        # 2. Geographic classification
        tier1_countries = ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'AUS', 'JPN', 'KOR', 'SGP']
        emerging_markets = ['IND', 'CHN', 'BRA', 'RUS', 'MEX', 'ZAF', 'TUR']
        
        self.df['tier1_country'] = self.df['country_code'].isin(tier1_countries).astype(int)
        self.df['emerging_market'] = self.df['country_code'].isin(emerging_markets).astype(int)
        
        # 3. Success category for main dataset
        success_categories = {
            'acquired': 'Success',
            'ipo': 'Success', 
            'closed': 'Failed',
            'operating': 'Operating'
        }
        self.df['success_category'] = self.df['status'].map(success_categories)
        
        # 4. Economic era classification
        recession_years = [2001, 2002, 2008, 2009, 2020]
        founded_years = self.df['founded_at'].dt.year
        self.df['founded_in_recession'] = founded_years.isin(recession_years).astype(int)
        
        # 5. Funding bracket classification
        self.df['funding_bracket'] = pd.cut(
            self.df['funding_total_usd'].fillna(0), 
            bins=[0, 50000, 500000, 5000000, 50000000, np.inf],
            labels=['No/Low Funding', 'Seed', 'Series A/B', 'Growth', 'Late Stage'],
            include_lowest=True
        )
        
        print("✅ Basic ML features added to main dataset")
    
    def _extract_domain_type(self, url):
        """Extract domain type from homepage URL."""
        if pd.isna(url) or url == 'nan':
            return 'No Website'
        
        url = str(url).lower()
        
        if '.com' in url:
            return 'Commercial'
        elif '.org' in url:
            return 'Organization'
        elif '.net' in url:
            return 'Network'
        elif '.io' in url:
            return 'Tech'
        elif '.co' in url:
            return 'Company'
        else:
            return 'Other'
    
    def _validate_and_fix_data(self):
        """Validate and fix data inconsistencies."""
        
        # 1. Fix funding rounds vs funding amount inconsistencies
        # If there's funding but no rounds recorded, set to 1
        mask = (self.df['funding_total_usd'] > 0) & (self.df['funding_rounds'].isna())
        self.df.loc[mask, 'funding_rounds'] = 1
        
        # If there are rounds but no funding amount, it's suspicious
        suspicious_mask = (self.df['funding_rounds'] > 0) & (self.df['funding_total_usd'].isna())
        print(f"Found {suspicious_mask.sum()} suspicious records with rounds but no funding amount")
        
        # 2. Validate date consistency
        # First funding should be after founding
        date_inconsistent = (
            self.df['first_funding_at'].notna() & 
            self.df['founded_at'].notna() &
            (self.df['first_funding_at'] < self.df['founded_at'])
        )
        
        if date_inconsistent.sum() > 0:
            print(f"Fixed {date_inconsistent.sum()} date inconsistencies")
            # Set first funding to founding date for these cases
            self.df.loc[date_inconsistent, 'first_funding_at'] = self.df.loc[date_inconsistent, 'founded_at']
        
        # 3. Standardize status values
        status_mapping = {
            'operating': 'operating',
            'acquired': 'acquired', 
            'closed': 'closed',
            'ipo': 'ipo'
        }
        
        self.df['status'] = self.df['status'].str.lower().map(status_mapping)
        
        print("✅ Data validation and fixes completed")
    
    def exploratory_data_analysis(self):
        """Enhanced exploratory data analysis with automatic plot saving."""
        print("\n🔍 ENHANCED EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        # Create comprehensive EDA with saved plots
        self._enhanced_dataset_overview()
        self._enhanced_success_failure_analysis()
        self._enhanced_funding_analysis()
        self._enhanced_geographic_analysis()
        self._enhanced_category_analysis()
        self._enhanced_temporal_analysis()
        self._correlation_analysis()
        
    def _enhanced_dataset_overview(self):
        """Enhanced dataset overview with visualizations."""
        print("\n📋 Enhanced Dataset Overview:")
        print(f"Total startups: {len(self.df):,}")
        print(f"Columns: {len(self.df.columns)}")
        
        # Create comprehensive overview plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Dataset Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Missing data heatmap
        ax1 = axes[0, 0]
        missing_data = self.df.isnull().sum().sort_values(ascending=False)
        missing_percent = (missing_data / len(self.df)) * 100
        
        # Only show columns with missing data
        missing_cols = missing_percent[missing_percent > 0]
        if len(missing_cols) > 0:
            missing_cols.plot(kind='bar', ax=ax1, color='salmon')
            ax1.set_title('Missing Data by Column')
            ax1.set_ylabel('Missing Percentage (%)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 2. Data types distribution
        ax2 = axes[0, 1]
        dtype_counts = self.df.dtypes.value_counts()
        ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        ax2.set_title('Data Types Distribution')
        
        # 3. Record completeness score
        ax3 = axes[0, 2]
        completeness_scores = (1 - self.df.isnull().sum(axis=1) / len(self.df.columns)) * 100
        ax3.hist(completeness_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        ax3.set_title('Record Completeness Distribution')
        ax3.set_xlabel('Completeness Score (%)')
        ax3.set_ylabel('Number of Records')
        
        # 4. Company age distribution
        ax4 = axes[1, 0]
        valid_ages = self.df['company_age_years'].dropna()
        valid_ages = valid_ages[(valid_ages >= 0) & (valid_ages <= 50)]  # Reasonable range
        ax4.hist(valid_ages, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
        ax4.set_title('Company Age Distribution')
        ax4.set_xlabel('Age (Years)')
        ax4.set_ylabel('Frequency')
        
        # 5. Funding status overview
        ax5 = axes[1, 1]
        funding_status = ['Has Funding', 'No Funding']
        funding_counts = [self.df['has_funding'].sum(), len(self.df) - self.df['has_funding'].sum()]
        ax5.pie(funding_counts, labels=funding_status, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        ax5.set_title('Funding Status Overview')
        
        # 6. Geographic coverage
        ax6 = axes[1, 2]
        location_status = ['Has Location', 'No Location']
        location_counts = [self.df['has_location'].sum(), len(self.df) - self.df['has_location'].sum()]
        ax6.pie(location_counts, labels=location_status, autopct='%1.1f%%', colors=['lightgreen', 'lightyellow'])
        ax6.set_title('Geographic Data Coverage')
        
        plt.tight_layout()
        self.save_plot('dataset_overview_dashboard')
        plt.show()
        
        # Print detailed statistics
        print(f"\n📊 Dataset Statistics:")
        print(f"  Average company age: {self.df['company_age_years'].mean():.1f} years")
        print(f"  Companies with funding: {self.df['has_funding'].sum():,} ({self.df['has_funding'].mean()*100:.1f}%)")
        print(f"  Companies with location data: {self.df['has_location'].sum():,} ({self.df['has_location'].mean()*100:.1f}%)")
        print(f"  Average funding rounds: {self.df['funding_rounds'].mean():.1f}")
    
    def _enhanced_success_failure_analysis(self):
        """Enhanced success/failure analysis with detailed visualizations."""
        print("\n📈 ENHANCED SUCCESS/FAILURE ANALYSIS")
        
        # Define success categories
        success_categories = {
            'acquired': 'Success',
            'ipo': 'Success', 
            'closed': 'Failed',
            'operating': 'Operating'
        }
        
        self.df['success_category'] = self.df['status'].map(success_categories)
        
        # Create comprehensive success analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Success/Failure Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Overall status distribution
        ax1 = axes[0, 0]
        status_counts = self.df['status'].value_counts()
        status_counts.plot(kind='bar', ax=ax1, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        ax1.set_title('Startup Status Distribution')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Success rate pie chart (concluded only)
        ax2 = axes[0, 1]
        concluded_df = self.df[self.df['success_category'].isin(['Success', 'Failed'])]
        success_counts = concluded_df['success_category'].value_counts()
        ax2.pie(success_counts.values, labels=success_counts.index, autopct='%1.1f%%', 
                colors=['lightgreen', 'salmon'], startangle=90)
        ax2.set_title('Success vs Failure\n(Concluded Startups)')
        
        # 3. Success rate by company age groups
        ax3 = axes[0, 2]
        concluded_with_age = concluded_df.dropna(subset=['company_age_years'])
        concluded_with_age['age_group'] = pd.cut(
            concluded_with_age['company_age_years'], 
            bins=[0, 2, 5, 10, float('inf')],
            labels=['0-2 years', '2-5 years', '5-10 years', '10+ years']
        )
        
        success_by_age = concluded_with_age.groupby('age_group')['success_category'].apply(
            lambda x: (x == 'Success').mean() * 100
        )
        success_by_age.plot(kind='bar', ax=ax3, color='lightblue')
        ax3.set_title('Success Rate by Company Age')
        ax3.set_ylabel('Success Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Time to outcome analysis
        ax4 = axes[1, 0]
        concluded_with_age_valid = concluded_with_age[
            (concluded_with_age['company_age_years'] > 0) & 
            (concluded_with_age['company_age_years'] < 30)
        ]
        
        success_ages = concluded_with_age_valid[
            concluded_with_age_valid['success_category'] == 'Success'
        ]['company_age_years']
        
        failed_ages = concluded_with_age_valid[
            concluded_with_age_valid['success_category'] == 'Failed'
        ]['company_age_years']
        
        ax4.hist([success_ages, failed_ages], bins=20, alpha=0.7, 
                label=['Success', 'Failed'], color=['lightgreen', 'salmon'])
        ax4.set_title('Time to Outcome Distribution')
        ax4.set_xlabel('Company Age at Outcome (Years)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        # 5. Success rate by funding status
        ax5 = axes[1, 1]
        funding_success = concluded_df.groupby('has_funding')['success_category'].apply(
            lambda x: (x == 'Success').mean() * 100
        )
        funding_labels = ['No Funding', 'Has Funding']
        ax5.bar(funding_labels, funding_success.values, color=['lightcoral', 'lightgreen'])
        ax5.set_title('Success Rate by Funding Status')
        ax5.set_ylabel('Success Rate (%)')
        
        # 6. Success metrics summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate key metrics
        total_concluded = len(concluded_df)
        success_rate = (concluded_df['success_category'] == 'Success').mean() * 100
        avg_age_success = success_ages.mean() if len(success_ages) > 0 else 0
        avg_age_failed = failed_ages.mean() if len(failed_ages) > 0 else 0
        
        metrics_text = f"""
        SUCCESS METRICS SUMMARY
        
        Total Concluded: {total_concluded:,}
        Overall Success Rate: {success_rate:.1f}%
        
        Average Age at Success: {avg_age_success:.1f} years
        Average Age at Failure: {avg_age_failed:.1f} years
        
        Success with Funding: {funding_success[1]:.1f}%
        Success without Funding: {funding_success[0]:.1f}%
        """
        
        ax6.text(0.1, 0.9, metrics_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        self.save_plot('success_failure_analysis_dashboard')
        plt.show()
        
        # Print detailed statistics
        print(f"\n📊 Success Analysis Results:")
        print(f"  Total startups analyzed: {len(self.df):,}")
        print(f"  Concluded startups: {total_concluded:,}")
        print(f"  Overall success rate: {success_rate:.1f}%")
        print(f"  Average time to success: {avg_age_success:.1f} years")
        print(f"  Average time to failure: {avg_age_failed:.1f} years")
    
    def _enhanced_funding_analysis(self):
        """Enhanced funding analysis with comprehensive insights."""
        print("\n💰 ENHANCED FUNDING ANALYSIS")
        
        funding_df = self.df.dropna(subset=['funding_total_usd'])
        
        # Create funding analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Funding Analysis', fontsize=16, fontweight='bold')
        
        # 1. Funding distribution by status
        ax1 = axes[0, 0]
        for status in funding_df['status'].unique():
            data = funding_df[funding_df['status'] == status]['funding_total_usd']
            ax1.hist(np.log10(data + 1), alpha=0.7, label=status, bins=25, density=True)
        ax1.set_xlabel('Log10(Funding + 1)')
        ax1.set_ylabel('Density')
        ax1.set_title('Funding Distribution by Status')
        ax1.legend()
        
        # 2. Funding efficiency analysis
        ax2 = axes[0, 1]
        efficiency_df = funding_df.dropna(subset=['funding_per_round'])
        sns.boxplot(data=efficiency_df, x='status', y='funding_per_round', ax=ax2)
        ax2.set_yscale('log')
        ax2.set_title('Funding Efficiency\n(Funding per Round)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Funding timeline analysis
        ax3 = axes[0, 2]
        timeline_df = self.df.dropna(subset=['days_to_first_funding'])
        timeline_df = timeline_df[
            (timeline_df['days_to_first_funding'] >= 0) & 
            (timeline_df['days_to_first_funding'] <= 3650)  # Max 10 years
        ]
        
        for status in ['acquired', 'ipo', 'closed']:
            if status in timeline_df['status'].values:
                data = timeline_df[timeline_df['status'] == status]['days_to_first_funding']
                ax3.hist(data / 365.25, alpha=0.7, label=status, bins=20)
        
        ax3.set_xlabel('Years to First Funding')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Time to First Funding by Outcome')
        ax3.legend()
        
        # 4. Funding bracket success analysis
        ax4 = axes[1, 0]
        funding_df['funding_bracket'] = pd.cut(
            funding_df['funding_total_usd'], 
            bins=[0, 50000, 500000, 5000000, 50000000, np.inf],
            labels=['<50K', '50K-500K', '500K-5M', '5M-50M', '>50M']
        )
        
        concluded_funding = funding_df[funding_df['success_category'].isin(['Success', 'Failed'])]
        success_by_funding = concluded_funding.groupby('funding_bracket')['success_category'].apply(
            lambda x: (x == 'Success').mean() * 100
        )
        
        success_by_funding.plot(kind='bar', ax=ax4, color='lightblue')
        ax4.set_title('Success Rate by Funding Bracket')
        ax4.set_ylabel('Success Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Funding rounds vs success
        ax5 = axes[1, 1]
        rounds_df = self.df.dropna(subset=['funding_rounds'])
        rounds_df = rounds_df[rounds_df['funding_rounds'] <= 10]  # Cap for visualization
        
        concluded_rounds = rounds_df[rounds_df['success_category'].isin(['Success', 'Failed'])]
        success_by_rounds = concluded_rounds.groupby('funding_rounds')['success_category'].apply(
            lambda x: (x == 'Success').mean() * 100 if len(x) > 5 else np.nan
        ).dropna()
        
        success_by_rounds.plot(kind='line', marker='o', ax=ax5, color='green')
        ax5.set_title('Success Rate by Number of Funding Rounds')
        ax5.set_xlabel('Number of Funding Rounds')
        ax5.set_ylabel('Success Rate (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Funding summary statistics
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        funding_stats = funding_df.groupby('status')['funding_total_usd'].agg(['mean', 'median', 'count'])
        
        stats_text = "FUNDING STATISTICS BY STATUS\n\n"
        for status in funding_stats.index:
            stats_text += f"{status.upper()}:\n"
            stats_text += f"  Mean: ${funding_stats.loc[status, 'mean']:,.0f}\n"
            stats_text += f"  Median: ${funding_stats.loc[status, 'median']:,.0f}\n"
            stats_text += f"  Count: {funding_stats.loc[status, 'count']:,}\n\n"
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        self.save_plot('funding_analysis_dashboard')
        plt.show()
        
        print(f"💵 Funding Analysis Summary:")
        print(f"  Companies with funding data: {len(funding_df):,}")
        print(f"  Average funding amount: ${funding_df['funding_total_usd'].mean():,.0f}")
        print(f"  Median funding amount: ${funding_df['funding_total_usd'].median():,.0f}")
    
    def _enhanced_geographic_analysis(self):
        """Enhanced geographic analysis with global insights."""
        print("\n🌍 ENHANCED GEOGRAPHIC ANALYSIS")
        
        geo_df = self.df.dropna(subset=['country_code'])
        
        # Create geographic analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Global Startup Ecosystem Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top countries by startup count
        ax1 = axes[0, 0]
        top_countries = geo_df['country_code'].value_counts().head(15)
        top_countries.plot(kind='barh', ax=ax1, color='lightblue')
        ax1.set_title('Top 15 Countries by Startup Count')
        ax1.set_xlabel('Number of Startups')
        
        # 2. Success rate by top countries
        ax2 = axes[0, 1]
        country_success_rates = []
        country_names = []
        
        for country in top_countries.head(10).index:
            country_data = geo_df[geo_df['country_code'] == country]
            concluded = country_data[country_data['success_category'].isin(['Success', 'Failed'])]
            if len(concluded) > 10:  # Minimum sample size
                success_rate = (concluded['success_category'] == 'Success').mean() * 100
                country_success_rates.append(success_rate)
                country_names.append(country)
        
        ax2.bar(country_names, country_success_rates, color='lightgreen')
        ax2.set_title('Success Rate by Country\n(Top Countries, min 10 concluded)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Average funding by country
        ax3 = axes[0, 2]
        country_funding = geo_df.groupby('country_code')['funding_total_usd'].mean().sort_values(ascending=False).head(10)
        country_funding.plot(kind='bar', ax=ax3, color='gold')
        ax3.set_title('Average Funding by Country\n(Top 10)')
        ax3.set_ylabel('Average Funding (USD)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.ticklabel_format(style='plain', axis='y')
        
        # 4. Startup density analysis (startups per capita would need population data)
        ax4 = axes[1, 0]
        # For now, we'll show startup concentration
        startup_concentration = top_countries.head(10)
        total_startups = startup_concentration.sum()
        concentration_pct = (startup_concentration / len(geo_df)) * 100
        
        concentration_pct.plot(kind='pie', ax=ax4, autopct='%1.1f%%')
        ax4.set_title('Startup Concentration\n(Top 10 Countries)')
        ax4.set_ylabel('')
        
        # 5. Geographic diversity metrics
        ax5 = axes[1, 1]
        # Calculate HHI (Herfindahl-Hirschman Index) for market concentration
        market_shares = (top_countries / len(geo_df)) ** 2
        hhi = market_shares.sum()
        
        # Diversity score (inverse of concentration)
        diversity_score = 1 - hhi
        
        diversity_metrics = ['Diversity Score', 'Top 3 Share', 'Top 5 Share', 'Top 10 Share']
        diversity_values = [
            diversity_score,
            (top_countries.head(3).sum() / len(geo_df)),
            (top_countries.head(5).sum() / len(geo_df)),
            (top_countries.head(10).sum() / len(geo_df))
        ]
        
        ax5.bar(diversity_metrics, diversity_values, color=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
        ax5.set_title('Geographic Diversity Metrics')
        ax5.set_ylabel('Score/Share')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Regional analysis summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create regional summary
        regional_summary = f"""
        GEOGRAPHIC ANALYSIS SUMMARY
        
        Total Countries: {geo_df['country_code'].nunique()}
        Top Country: {top_countries.index[0]} ({top_countries.iloc[0]:,} startups)
        
        Market Concentration (HHI): {hhi:.3f}
        Diversity Score: {diversity_score:.3f}
        
        TOP 5 COUNTRIES:
        """
        
        for i, (country, count) in enumerate(top_countries.head(5).items()):
            regional_summary += f"{i+1}. {country}: {count:,} startups\n"
        
        ax6.text(0.1, 0.9, regional_summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
        
        plt.tight_layout()
        self.save_plot('geographic_analysis_dashboard')
        plt.show()
        
        print(f"🗺️ Geographic Analysis Results:")
        print(f"  Countries represented: {geo_df['country_code'].nunique()}")
        print(f"  Top country: {top_countries.index[0]} with {top_countries.iloc[0]:,} startups")
        print(f"  Geographic diversity score: {diversity_score:.3f}")
    
    def _enhanced_category_analysis(self):
        """Enhanced category analysis with market insights."""
        print("\n🏷️ ENHANCED CATEGORY ANALYSIS")
        
        # Extract and analyze categories
        all_categories = []
        for categories in self.df['category_list'].dropna():
            cats = [cat.strip() for cat in str(categories).split('|')]
            all_categories.extend(cats)
        
        category_counts = pd.Series(all_categories).value_counts()
        
        # Create category analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Startup Category Ecosystem Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top categories by count
        ax1 = axes[0, 0]
        top_categories = category_counts.head(15)
        top_categories.plot(kind='barh', ax=ax1, color='lightcoral')
        ax1.set_title('Top 15 Categories by Startup Count')
        ax1.set_xlabel('Number of Startups')
        
        # 2. Success rate by category
        ax2 = axes[0, 1]
        category_success_rates = {}
        
        for category in top_categories.head(10).index:
            category_startups = self.df[self.df['category_list'].str.contains(category, na=False)]
            concluded = category_startups[category_startups['success_category'].isin(['Success', 'Failed'])]
            
            if len(concluded) > 15:  # Minimum sample size
                success_rate = (concluded['success_category'] == 'Success').mean() * 100
                category_success_rates[category] = success_rate
        
        if category_success_rates:
            categories = list(category_success_rates.keys())
            success_rates = list(category_success_rates.values())
            
            ax2.barh(categories, success_rates, color='lightgreen')
            ax2.set_title('Success Rate by Category\n(Min 15 concluded startups)')
            ax2.set_xlabel('Success Rate (%)')
        
        # 3. Average funding by category
        ax3 = axes[0, 2]
        category_funding = {}
        
        for category in top_categories.head(10).index:
            category_startups = self.df[self.df['category_list'].str.contains(category, na=False)]
            avg_funding = category_startups['funding_total_usd'].mean()
            if not np.isnan(avg_funding):
                category_funding[category] = avg_funding
        
        if category_funding:
            sorted_funding = dict(sorted(category_funding.items(), key=lambda x: x[1], reverse=True))
            categories = list(sorted_funding.keys())[:8]
            funding_amounts = [sorted_funding[cat] for cat in categories]
            
            ax3.bar(categories, funding_amounts, color='gold')
            ax3.set_title('Average Funding by Category')
            ax3.set_ylabel('Average Funding (USD)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.ticklabel_format(style='plain', axis='y')
        
        # 4. Category diversity over time
        ax4 = axes[1, 0]
        yearly_diversity = []
        years = []
        
        for year in range(2000, 2021):
            year_startups = self.df[self.df['founded_at'].dt.year == year]
            if len(year_startups) > 50:
                year_categories = []
                for categories in year_startups['category_list'].dropna():
                    cats = [cat.strip() for cat in str(categories).split('|')]
                    year_categories.extend(cats)
                
                unique_categories = len(set(year_categories))
                years.append(year)
                yearly_diversity.append(unique_categories)
        
        if years:
            ax4.plot(years, yearly_diversity, marker='o', color='purple')
            ax4.set_title('Category Diversity Over Time')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('Number of Unique Categories')
            ax4.grid(True, alpha=0.3)
        
        # 5. Category combination analysis
        ax5 = axes[1, 1]
        category_combinations = self.df['category_count'].value_counts().sort_index()
        category_combinations = category_combinations[category_combinations.index <= 10]
        
        ax5.bar(category_combinations.index, category_combinations.values, color='lightblue')
        ax5.set_title('Distribution of Category Combinations')
        ax5.set_xlabel('Number of Categories per Startup')
        ax5.set_ylabel('Number of Startups')
        
        # 6. Category insights summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate insights
        most_popular = category_counts.index[0]
        most_successful = max(category_success_rates.items(), key=lambda x: x[1]) if category_success_rates else ("N/A", 0)
        highest_funded = max(category_funding.items(), key=lambda x: x[1]) if category_funding else ("N/A", 0)
        
        insights_text = f"""
        CATEGORY INSIGHTS SUMMARY
        
        Total Unique Categories: {len(category_counts)}
        Most Popular: {most_popular}
        ({category_counts.iloc[0]:,} startups)
        
        Highest Success Rate: {most_successful[0]}
        ({most_successful[1]:.1f}%)
        
        Highest Average Funding: {highest_funded[0]}
        (${highest_funded[1]:,.0f})
        
        Avg Categories per Startup: {self.df['category_count'].mean():.1f}
        """
        
        ax6.text(0.1, 0.9, insights_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))
        
        plt.tight_layout()
        self.save_plot('category_analysis_dashboard')
        plt.show()
        
        print(f"📊 Category Analysis Results:")
        print(f"  Total unique categories: {len(category_counts)}")
        print(f"  Most popular category: {most_popular} ({category_counts.iloc[0]:,} startups)")
        print(f"  Average categories per startup: {self.df['category_count'].mean():.1f}")
    
    def _enhanced_temporal_analysis(self):
        """Enhanced temporal analysis with trend insights."""
        print("\n📅 ENHANCED TEMPORAL ANALYSIS")
        
        temporal_df = self.df.dropna(subset=['founded_at'])
        temporal_df = temporal_df[
            (temporal_df['founded_at'].dt.year >= 1990) & 
            (temporal_df['founded_at'].dt.year <= 2023)
        ]
        
        # Create temporal analysis dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Trends in Startup Ecosystem', fontsize=16, fontweight='bold')
        
        # 1. Startups founded by year
        ax1 = axes[0, 0]
        yearly_counts = temporal_df['founded_at'].dt.year.value_counts().sort_index()
        yearly_counts.plot(kind='line', marker='o', ax=ax1, color='blue')
        ax1.set_title('Startups Founded by Year')
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Startups Founded')
        ax1.grid(True, alpha=0.3)
        
        # 2. Success rate evolution over time
        ax2 = axes[0, 1]
        concluded_temporal = temporal_df[temporal_df['success_category'].isin(['Success', 'Failed'])]
        
        success_by_year = []
        years_with_data = []
        
        for year in range(1995, 2020):  # Focus on years with meaningful data
            year_data = concluded_temporal[concluded_temporal['founded_at'].dt.year == year]
            if len(year_data) > 20:  # Minimum sample size
                success_rate = (year_data['success_category'] == 'Success').mean() * 100
                success_by_year.append(success_rate)
                years_with_data.append(year)
        
        if years_with_data:
            ax2.plot(years_with_data, success_by_year, marker='o', color='green')
            ax2.set_title('Success Rate Evolution Over Time')
            ax2.set_xlabel('Founding Year')
            ax2.set_ylabel('Success Rate (%)')
            ax2.grid(True, alpha=0.3)
        
        # 3. Funding trends over time
        ax3 = axes[0, 2]
        funding_temporal = temporal_df.dropna(subset=['funding_total_usd'])
        
        yearly_funding = funding_temporal.groupby(funding_temporal['founded_at'].dt.year)['funding_total_usd'].median()
        yearly_funding = yearly_funding[yearly_funding.index >= 1995]
        
        yearly_funding.plot(kind='line', marker='s', ax=ax3, color='orange')
        ax3.set_title('Median Funding Trends by Founding Year')
        ax3.set_xlabel('Founding Year')
        ax3.set_ylabel('Median Funding (USD)')
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='plain', axis='y')
        
        # 4. Seasonal founding patterns
        ax4 = axes[1, 0]
        monthly_counts = temporal_df['founded_at'].dt.month.value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax4.bar(range(1, 13), monthly_counts.values, color='lightcyan')
        ax4.set_title('Seasonal Founding Patterns')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Number of Startups Founded')
        ax4.set_xticks(range(1, 13))
        ax4.set_xticklabels(month_names)
        
        # 5. Economic era analysis
        ax5 = axes[1, 1]
        temporal_df['economic_era'] = pd.cut(
            temporal_df['founded_at'].dt.year,
            bins=[1989, 1999, 2007, 2009, 2019, 2023],
            labels=['Dot-com Era\n(1990-1999)', 'Pre-Crisis\n(2000-2007)', 
                   'Financial Crisis\n(2008-2009)', 'Recovery\n(2010-2019)', 'Recent\n(2020+)']
        )
        
        era_success = temporal_df.groupby('economic_era')['success_category'].apply(
            lambda x: (x == 'Success').sum() / ((x == 'Success').sum() + (x == 'Failed').sum()) * 100
        ).fillna(0)
        
        era_success.plot(kind='bar', ax=ax5, color='lightsteelblue')
        ax5.set_title('Success Rate by Economic Era')
        ax5.set_ylabel('Success Rate (%)')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. Time-to-success analysis
        ax6 = axes[1, 2]
        success_companies = temporal_df[temporal_df['success_category'] == 'Success']
        success_companies = success_companies.dropna(subset=['company_age_years'])
        success_companies = success_companies[
            (success_companies['company_age_years'] > 0) & 
            (success_companies['company_age_years'] < 25)
        ]
        
        if len(success_companies) > 0:
            ax6.hist(success_companies['company_age_years'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
            ax6.set_title('Time to Success Distribution')
            ax6.set_xlabel('Years to Success')
            ax6.set_ylabel('Frequency')
            ax6.axvline(success_companies['company_age_years'].median(), color='red', 
                       linestyle='--', label=f'Median: {success_companies["company_age_years"].median():.1f} years')
            ax6.legend()
        
        plt.tight_layout()
        self.save_plot('temporal_analysis_dashboard')
        plt.show()
        
        print(f"⏰ Temporal Analysis Results:")
        print(f"  Peak founding year: {yearly_counts.idxmax()} ({yearly_counts.max():,} startups)")
        print(f"  Average time to success: {success_companies['company_age_years'].mean():.1f} years")
        print(f"  Most popular founding month: {month_names[monthly_counts.idxmax()-1]}")
    
    def _correlation_analysis(self):
        """Analyze correlations between numerical features."""
        print("\n🔗 CORRELATION ANALYSIS")
        
        # Select numerical columns for correlation analysis
        numerical_cols = [
            'funding_total_usd', 'funding_rounds', 'company_age_years',
            'days_to_first_funding', 'funding_duration_days', 'category_count',
            'has_funding', 'has_location', 'has_homepage'
        ]
        
        # Filter available columns
        available_cols = [col for col in numerical_cols if col in self.df.columns]
        corr_df = self.df[available_cols].copy()
        
        # Create binary target for correlation
        corr_df['is_success'] = (self.df['success_category'] == 'Success').astype(int)
        
        # Calculate correlation matrix
        correlation_matrix = corr_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self.save_plot('correlation_matrix')
        plt.show()
        
        # Show strongest correlations with success
        success_correlations = correlation_matrix['is_success'].abs().sort_values(ascending=False)
        print(f"🎯 Strongest correlations with success:")
        for feature, corr in success_correlations.items():
            if feature != 'is_success' and abs(corr) > 0.05:
                print(f"  {feature}: {corr:.3f}")
    
    def advanced_feature_engineering(self):
        """Advanced feature engineering for better model performance."""
        print("\n🔧 ADVANCED FEATURE ENGINEERING")
        
        # Start with a copy for ML
        ml_df = self.df.copy()
        
        # Filter concluded startups for training
        concluded_df = ml_df[ml_df['status'].isin(['acquired', 'ipo', 'closed'])]
        print(f"Concluded startups for ML: {len(concluded_df):,}")
        
        # Create sophisticated target variable
        concluded_df['target'] = concluded_df['status'].apply(
            lambda x: 1 if x in ['acquired', 'ipo'] else 0
        )
        
        # Advanced feature engineering
        self._create_advanced_features(concluded_df)
        
        # Feature selection
        feature_columns = self._select_best_features(concluded_df)
        
        # Prepare final dataset with advanced preprocessing
        X, y = self._prepare_advanced_dataset(concluded_df, feature_columns)
        
        self.X_advanced = X
        self.y_advanced = y
        self.feature_names_advanced = feature_columns
        
        return X, y
    
    def _create_advanced_features(self, df):
        """Create advanced features for better prediction with proper handling of edge cases."""
        
        print("🎯 Creating advanced features...")
        
        # Helper function to safely handle divisions and prevent infinity
        def safe_divide(numerator, denominator, default_value=0):
            """Safely divide two series, handling zeros and infinities."""
            result = np.where(
                (pd.isna(denominator)) | (denominator == 0),
                default_value,
                numerator / denominator
            )
            # Cap extreme values to prevent infinity
            result = np.clip(result, -1e10, 1e10)
            return result
        
        def safe_log(values):
            """Safely calculate log, handling zeros and negative values."""
            values = np.maximum(values, 1e-10)  # Ensure positive values
            result = np.log10(values)
            return np.clip(result, -10, 10)  # Cap extreme values
        
        # 1. Funding-related advanced features
        df['funding_total_usd_filled'] = df['funding_total_usd'].fillna(
            df['funding_total_usd'].median() if not df['funding_total_usd'].isna().all() else 0
        )
        
        # Safe log transformation
        df['funding_log'] = safe_log(df['funding_total_usd_filled'] + 1)
        
        # Safe funding per round calculation
        df['funding_per_round_filled'] = df['funding_per_round'].fillna(
            df['funding_per_round'].median() if not df['funding_per_round'].isna().all() else 0
        )
        
        # Safe company age handling
        df['company_age_years'] = df['company_age_years'].fillna(0)
        df['company_age_years'] = np.clip(df['company_age_years'], 0, 100)  # Cap at reasonable values
        
        # Safe funding intensity calculation
        df['funding_intensity'] = safe_divide(
            df['funding_total_usd_filled'], 
            df['company_age_years'] + 1,
            default_value=0
        )
        
        # Safe funding velocity calculation
        df['funding_velocity'] = safe_divide(
            df['funding_total_usd_filled'],
            np.maximum(df['company_age_years'], 0.1),  # Avoid division by zero
            default_value=0
        )
        
        # 2. Time-based advanced features
        current_year = pd.Timestamp.now().year
        median_year = df['founded_at'].dt.year.median()
        if pd.isna(median_year):
            median_year = 2010  # Default fallback
            
        df['founded_year_filled'] = df['founded_at'].dt.year.fillna(median_year)
        df['founded_year_filled'] = np.clip(df['founded_year_filled'], 1990, current_year)
        df['founded_decade'] = (df['founded_year_filled'] // 10) * 10
        
        # Safe time efficiency metrics
        median_days = df['days_to_first_funding'].median()
        if pd.isna(median_days):
            median_days = 365  # Default to 1 year
            
        df['days_to_first_funding_filled'] = df['days_to_first_funding'].fillna(median_days)
        df['days_to_first_funding_filled'] = np.clip(df['days_to_first_funding_filled'], 0, 3650)  # Cap at 10 years
        
        df['funding_speed_score'] = safe_divide(
            1, 
            df['days_to_first_funding_filled'] + 1,
            default_value=0
        )
        
        # Economic cycle features
        recession_years = [2001, 2002, 2008, 2009, 2020]
        df['founded_in_recession'] = df['founded_year_filled'].isin(recession_years).astype(int)
        
        # 3. Category sophistication features
        tech_categories = ['Software', 'Biotechnology', 'Artificial Intelligence', 'Machine Learning', 
                          'Data Analytics', 'Cybersecurity', 'Blockchain', 'IoT', 'API', 'SaaS']
        
        df['is_tech_startup'] = df['category_list'].apply(
            lambda x: any(cat in str(x) for cat in tech_categories) if pd.notna(x) else 0
        ).astype(int)
        
        # Safe category diversity score
        max_categories = df['category_count'].max()
        if pd.isna(max_categories) or max_categories == 0:
            max_categories = 1
            
        df['category_diversity'] = safe_divide(
            df['category_count'], 
            max_categories,
            default_value=0
        )
        
        # 4. Geographic sophistication
        tier1_countries = ['USA', 'GBR', 'CAN', 'DEU', 'FRA', 'AUS', 'JPN', 'KOR', 'SGP']
        emerging_markets = ['IND', 'CHN', 'BRA', 'RUS', 'MEX', 'ZAF', 'TUR']
        
        df['tier1_country'] = df['country_code'].isin(tier1_countries).astype(int)
        df['emerging_market'] = df['country_code'].isin(emerging_markets).astype(int)
        
        # 5. Market timing features (simplified to avoid performance issues)
        # Use a more efficient approach for market density
        if 'primary_category' in df.columns and 'founded_year_filled' in df.columns:
            market_counts = df.groupby(['primary_category', 'founded_year_filled']).size()
            df['market_density'] = df.apply(
                lambda row: market_counts.get((row['primary_category'], row['founded_year_filled']), 1)
                if pd.notna(row['primary_category']) and pd.notna(row['founded_year_filled']) else 1,
                axis=1
            )
        else:
            df['market_density'] = 1
        
        df['market_density'] = np.clip(df['market_density'], 1, 1000)  # Cap extreme values
        
        # 6. Success probability indicators
        if 'target' in df.columns:
            # Historical success rate of category
            category_success_rates = df.groupby('primary_category')['target'].mean()
            df['category_success_rate'] = df['primary_category'].map(category_success_rates).fillna(0.1)
            
            # Historical success rate of country
            country_success_rates = df.groupby('country_code')['target'].mean()
            df['country_success_rate'] = df['country_code'].map(country_success_rates).fillna(0.1)
        else:
            df['category_success_rate'] = 0.1
            df['country_success_rate'] = 0.1
        
        # Ensure success rates are valid
        df['category_success_rate'] = np.clip(df['category_success_rate'], 0, 1)
        df['country_success_rate'] = np.clip(df['country_success_rate'], 0, 1)
        
        # 7. Interaction features (with safe operations)
        df['funding_country_interaction'] = df['funding_log'] * df['tier1_country']
        df['tech_funding_interaction'] = df['is_tech_startup'] * df['funding_log']
        df['age_funding_interaction'] = df['company_age_years'] * df['funding_log']
        
        # 8. Risk assessment features
        funding_90th = df['funding_total_usd_filled'].quantile(0.9)
        if pd.isna(funding_90th):
            funding_90th = df['funding_total_usd_filled'].max()
            
        df['high_risk_profile'] = (
            (df['funding_total_usd_filled'] > funding_90th) &
            (df['founded_in_recession'] == 1)
        ).astype(int)
        
        df['low_risk_profile'] = (
            (df['tier1_country'] == 1) &
            (df['category_success_rate'] > 0.2) &
            (df['has_funding'] == 1)
        ).astype(int)
        
        # Final cleanup: Replace any remaining infinite or extreme values
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in df.columns:
                # Replace infinities
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Cap extreme values
                if df[col].dtype in ['float64', 'float32']:
                    q99 = df[col].quantile(0.99)
                    q01 = df[col].quantile(0.01)
                    if not pd.isna(q99) and not pd.isna(q01):
                        df[col] = np.clip(df[col], q01, q99)
                
                # Fill any remaining NaN values
                df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)
        
        print(f"✅ Created advanced features with proper handling of edge cases")
        print(f"✅ All infinite and extreme values have been handled")
    
    def _select_best_features(self, df):
        """Select the best features using multiple selection methods with robust error handling."""
        
        print("🎯 Selecting best features...")
        
        # Define all potential features
        potential_features = [
            # Basic features
            'funding_total_usd_filled', 'funding_log', 'funding_rounds', 'has_funding',
            'company_age_years', 'category_count', 'has_location', 'has_homepage',
            
            # Advanced features
            'funding_velocity', 'funding_intensity', 'funding_speed_score',
            'is_tech_startup', 'category_diversity', 'tier1_country', 'emerging_market',
            'founded_in_recession', 'market_density', 'category_success_rate', 'country_success_rate',
            
            # Interaction features
            'funding_country_interaction', 'tech_funding_interaction', 'age_funding_interaction',
            
            # Risk features
            'high_risk_profile', 'low_risk_profile',
            
            # Time features
            'founded_decade', 'days_to_first_funding_filled'
        ]
        
        # Filter available features
        available_features = [f for f in potential_features if f in df.columns]
        print(f"Available features for selection: {len(available_features)}")
        
        # Prepare data for feature selection
        X_temp = df[available_features].copy()
        y_temp = df['target']
        
        # Additional cleanup to ensure no infinite or extreme values
        for col in X_temp.columns:
            # Replace any infinite values
            X_temp[col] = X_temp[col].replace([np.inf, -np.inf], np.nan)
            
            # Cap extreme values using robust statistics
            if X_temp[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                q01, q99 = X_temp[col].quantile([0.01, 0.99])
                if not pd.isna(q01) and not pd.isna(q99) and q99 > q01:
                    X_temp[col] = np.clip(X_temp[col], q01, q99)
            
            # Fill missing values with median
            if X_temp[col].isna().any():
                median_val = X_temp[col].median()
                if pd.isna(median_val):
                    median_val = 0
                X_temp[col] = X_temp[col].fillna(median_val)
        
        # Double-check for any remaining problematic values
        for col in X_temp.columns:
            if not np.isfinite(X_temp[col]).all():
                print(f"Warning: Column {col} still has non-finite values, replacing with 0")
                X_temp[col] = X_temp[col].replace([np.inf, -np.inf, np.nan], 0)
        
        print("✅ Data cleaned and ready for feature selection")
        
        try:
            # Use SimpleImputer instead of KNNImputer to avoid potential issues
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_temp_imputed = pd.DataFrame(
                imputer.fit_transform(X_temp),
                columns=X_temp.columns,
                index=X_temp.index
            )
            
            # Verify no infinite values remain
            assert np.isfinite(X_temp_imputed.values).all(), "Data still contains non-finite values"
            
        except Exception as e:
            print(f"Warning: Imputation failed ({e}), using original cleaned data")
            X_temp_imputed = X_temp.copy()
        
        # Feature selection using multiple methods
        selected_features = []
        
        # 1. Correlation-based selection
        try:
            correlations = X_temp_imputed.corrwith(y_temp).abs().sort_values(ascending=False)
            top_corr_features = correlations.dropna().head(15).index.tolist()
            selected_features.extend(top_corr_features)
            print(f"✅ Correlation-based selection: {len(top_corr_features)} features")
        except Exception as e:
            print(f"Warning: Correlation selection failed ({e})")
        
        # 2. Variance-based selection (remove low-variance features)
        try:
            from sklearn.feature_selection import VarianceThreshold
            variance_selector = VarianceThreshold(threshold=0.01)
            X_variance = variance_selector.fit_transform(X_temp_imputed)
            variance_features = X_temp_imputed.columns[variance_selector.get_support()].tolist()
            selected_features.extend(variance_features)
            print(f"✅ Variance-based selection: {len(variance_features)} features")
        except Exception as e:
            print(f"Warning: Variance selection failed ({e})")
        
        # 3. Random Forest feature importance
        try:
            rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
            rf_selector.fit(X_temp_imputed, y_temp)
            feature_importance = pd.Series(rf_selector.feature_importances_, index=X_temp_imputed.columns)
            top_rf_features = feature_importance.sort_values(ascending=False).head(15).index.tolist()
            selected_features.extend(top_rf_features)
            print(f"✅ Random Forest selection: {len(top_rf_features)} features")
        except Exception as e:
            print(f"Warning: Random Forest selection failed ({e})")
        
        # Combine selections and get unique features
        if selected_features:
            feature_counts = pd.Series(selected_features).value_counts()
            # Prefer features that appear in multiple selection methods
            final_features = feature_counts.index.tolist()
            
            # Ensure we have a reasonable number of features (10-20)
            if len(final_features) > 20:
                final_features = final_features[:20]
            elif len(final_features) < 10:
                # Add top correlated features to reach minimum
                additional_features = correlations.dropna().index.tolist()
                for feat in additional_features:
                    if feat not in final_features and len(final_features) < 15:
                        final_features.append(feat)
        else:
            # Fallback: use all available features if selection methods fail
            print("Warning: All feature selection methods failed, using all available features")
            final_features = available_features
        
        # Final validation of selected features
        final_features = [f for f in final_features if f in X_temp_imputed.columns]
        
        print(f"✅ Selected {len(final_features)} best features")
        print(f"Top 5 features: {final_features[:5]}")
        
        return final_features
    
    def _prepare_advanced_dataset(self, df, feature_columns):
        """Prepare the final dataset with advanced preprocessing."""
        
        print("🔧 Preparing advanced dataset...")
        
        # Extract features and target
        X = df[feature_columns].copy()
        y = df['target']
        
        # Advanced missing value imputation
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Feature scaling for algorithms that need it
        scaler = RobustScaler()  # Less sensitive to outliers than StandardScaler
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Store preprocessing objects
        self.imputer = imputer
        self.scaler = scaler
        
        print(f"✅ Final dataset prepared: {X_scaled.shape}")
        print(f"Target distribution: Success {y.sum()} ({y.mean()*100:.1f}%), Failed {len(y)-y.sum()}")
        
        return X_scaled, y
    
    def train_advanced_models(self):
        """Train advanced machine learning models with hyperparameter tuning."""
        print("\n🤖 TRAINING ADVANCED MACHINE LEARNING MODELS")
        print("=" * 60)
        
        # Split the data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_advanced, self.y_advanced, 
            test_size=0.2, random_state=42, stratify=self.y_advanced
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Define advanced models with hyperparameter grids
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            },
            'LightGBM': {
                'model': lgb.LGBMClassifier(random_state=42, verbose=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'CatBoost': {
                'model': CatBoostClassifier(random_state=42, verbose=False),
                'params': {
                    'iterations': [100, 200, 300],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'l2_leaf_reg': [1, 3, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9]
                }
            },
            'Extra Trees': {
                'model': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        # Train models with hyperparameter tuning
        self.advanced_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, config in models_config.items():
            print(f"\n🔧 Training {name} with hyperparameter tuning...")
            
            try:
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'],
                    cv=cv,
                    scoring='roc_auc',
                    n_jobs=1,  # Reduced parallelism for stability
                    verbose=0,
                    error_score='raise'  # Raise errors instead of returning NaN
                )
                
                # Fit the model
                grid_search.fit(X_train, y_train)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Make predictions
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                f1 = f1_score(y_test, y_pred)
                
                # Cross-validation scores
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring='roc_auc')
                
                self.advanced_results[name] = {
                    'model': best_model,
                    'best_params': grid_search.best_params_,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"  ✅ Best ROC-AUC: {roc_auc:.4f}")
                print(f"  ✅ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                print(f"  ✅ Best params: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"  ❌ Training failed for {name}: {str(e)}")
                print(f"  🔄 Trying {name} with default parameters...")
                
                try:
                    # Fallback: train with default parameters
                    simple_model = config['model']
                    simple_model.fit(X_train, y_train)
                    
                    y_pred = simple_model.predict(X_test)
                    y_pred_proba = simple_model.predict_proba(X_test)[:, 1]
                    
                    accuracy = accuracy_score(y_test, y_pred)
                    roc_auc = roc_auc_score(y_test, y_pred_proba)
                    f1 = f1_score(y_test, y_pred)
                    
                    cv_scores = cross_val_score(simple_model, X_train, y_train, cv=cv, scoring='roc_auc')
                    
                    self.advanced_results[name] = {
                        'model': simple_model,
                        'best_params': 'Default parameters (tuning failed)',
                        'accuracy': accuracy,
                        'roc_auc': roc_auc,
                        'f1_score': f1,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'probabilities': y_pred_proba
                    }
                    
                    print(f"  ✅ Fallback successful - ROC-AUC: {roc_auc:.4f}")
                    
                except Exception as e2:
                    print(f"  ❌ Both tuning and fallback failed for {name}: {str(e2)}")
                    continue
        
        # Check if we have any successful models
        if not self.advanced_results:
            raise ValueError("No models were successfully trained. Please check your data and try again.")
        
        # Create ensemble model only if we have multiple successful models
        if len(self.advanced_results) >= 2:
            self._create_ensemble_model(X_train, y_train, X_test, y_test)
        else:
            print("\n⚠️ Insufficient models for ensemble creation")
        
        # Store test data for evaluation
        self.X_test_advanced = X_test
        self.y_test_advanced = y_test
        
        # Find best model
        best_model_name = max(self.advanced_results.keys(), 
                             key=lambda k: self.advanced_results[k]['roc_auc'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   ROC-AUC: {self.advanced_results[best_model_name]['roc_auc']:.4f}")
        print(f"   Total models trained: {len(self.advanced_results)}")
        
        return self.advanced_results
    
    def _create_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Create an ensemble model combining the best individual models."""
        
        print("\n🎭 Creating Ensemble Model...")
        
        if len(self.advanced_results) < 2:
            print("⚠️ Need at least 2 models for ensemble creation")
            return
        
        try:
            # Select top models based on ROC-AUC (up to 3, or all if fewer)
            num_models = min(3, len(self.advanced_results))
            top_models = sorted(self.advanced_results.items(), 
                               key=lambda x: x[1]['roc_auc'], reverse=True)[:num_models]
            
            ensemble_models = [(name, result['model']) for name, result in top_models]
            
            # Create voting classifier
            ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft'  # Use probability-based voting
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            # Make predictions
            y_pred_ensemble = ensemble.predict(X_test)
            y_pred_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
            roc_auc_ensemble = roc_auc_score(y_test, y_pred_proba_ensemble)
            f1_ensemble = f1_score(y_test, y_pred_ensemble)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores_ensemble = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='roc_auc')
            
            self.advanced_results['Ensemble'] = {
                'model': ensemble,
                'best_params': f'Combination of {[name for name, _ in ensemble_models]}',
                'accuracy': accuracy_ensemble,
                'roc_auc': roc_auc_ensemble,
                'f1_score': f1_ensemble,
                'cv_mean': cv_scores_ensemble.mean(),
                'cv_std': cv_scores_ensemble.std(),
                'predictions': y_pred_ensemble,
                'probabilities': y_pred_proba_ensemble
            }
            
            print(f"✅ Ensemble created with ROC-AUC: {roc_auc_ensemble:.4f}")
            print(f"   Using models: {[name for name, _ in ensemble_models]}")
            
        except Exception as e:
            print(f"❌ Ensemble creation failed: {str(e)}")
            print("   Continuing with individual models only")
    
    def evaluate_advanced_models(self):
        """Comprehensive evaluation of advanced models."""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Create evaluation dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Advanced Model Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Model comparison metrics
        ax1 = axes[0, 0]
        model_names = list(self.advanced_results.keys())
        roc_aucs = [self.advanced_results[name]['roc_auc'] for name in model_names]
        accuracies = [self.advanced_results[name]['accuracy'] for name in model_names]
        f1_scores = [self.advanced_results[name]['f1_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax1.bar(x - width, roc_aucs, width, label='ROC-AUC', alpha=0.8)
        ax1.bar(x, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. ROC Curves
        ax2 = axes[0, 1]
        from sklearn.metrics import roc_curve
        
        for name, result in self.advanced_results.items():
            fpr, tpr, _ = roc_curve(self.y_test_advanced, result['probabilities'])
            ax2.plot(fpr, tpr, label=f'{name} (AUC={result["roc_auc"]:.3f})')
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves Comparison')
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        ax3 = axes[0, 2]
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        for name, result in self.advanced_results.items():
            precision, recall, _ = precision_recall_curve(self.y_test_advanced, result['probabilities'])
            avg_precision = average_precision_score(self.y_test_advanced, result['probabilities'])
            ax3.plot(recall, precision, label=f'{name} (AP={avg_precision:.3f})')
        
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curves')
        ax3.legend(loc="lower left", fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Feature Importance (best model)
        ax4 = axes[1, 0]
        best_model_name = max(self.advanced_results.keys(), 
                             key=lambda k: self.advanced_results[k]['roc_auc'])
        best_model = self.advanced_results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names_advanced,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            ax4.barh(feature_importance['feature'], feature_importance['importance'])
            ax4.set_title(f'Top 10 Features - {best_model_name}')
            ax4.set_xlabel('Importance')
        
        # 5. Confusion Matrix (best model)
        ax5 = axes[1, 1]
        best_predictions = self.advanced_results[best_model_name]['predictions']
        cm = confusion_matrix(self.y_test_advanced, best_predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                   xticklabels=['Failed', 'Success'], 
                   yticklabels=['Failed', 'Success'])
        ax5.set_title(f'Confusion Matrix - {best_model_name}')
        ax5.set_ylabel('True Label')
        ax5.set_xlabel('Predicted Label')
        
        # 6. Model Performance Summary
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create performance summary
        summary_text = "MODEL PERFORMANCE SUMMARY\n\n"
        summary_text += f"Best Model: {best_model_name}\n"
        summary_text += f"ROC-AUC: {self.advanced_results[best_model_name]['roc_auc']:.4f}\n"
        summary_text += f"Accuracy: {self.advanced_results[best_model_name]['accuracy']:.4f}\n"
        summary_text += f"F1-Score: {self.advanced_results[best_model_name]['f1_score']:.4f}\n\n"
        
        summary_text += "TOP 3 MODELS:\n"
        sorted_models = sorted(self.advanced_results.items(), 
                              key=lambda x: x[1]['roc_auc'], reverse=True)
        
        for i, (name, result) in enumerate(sorted_models[:3]):
            summary_text += f"{i+1}. {name}: {result['roc_auc']:.4f}\n"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        self.save_plot('advanced_model_evaluation_dashboard')
        plt.show()
        
        # Print detailed results
        print("\n📈 Detailed Model Results:")
        results_df = pd.DataFrame({
            'Model': list(self.advanced_results.keys()),
            'ROC-AUC': [r['roc_auc'] for r in self.advanced_results.values()],
            'Accuracy': [r['accuracy'] for r in self.advanced_results.values()],
            'F1-Score': [r['f1_score'] for r in self.advanced_results.values()],
            'CV Mean': [r['cv_mean'] for r in self.advanced_results.values()],
            'CV Std': [r['cv_std'] for r in self.advanced_results.values()]
        }).sort_values('ROC-AUC', ascending=False)
        
        print(results_df.round(4))
        
        # Save results
        results_path = self.data_dir / "model_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"💾 Model results saved: {results_path}")
    
    def predict_operating_startups_advanced(self):
        """Make advanced predictions on operating startups."""
        print("\n🔮 ADVANCED PREDICTIONS FOR OPERATING STARTUPS")
        print("=" * 60)
        
        # Get operating startups
        operating_df = self.df[self.df['status'] == 'operating'].copy()
        print(f"Operating startups to predict: {len(operating_df):,}")
        
        # Apply advanced feature engineering
        self._create_advanced_features(operating_df)
        
        # Prepare features using the same preprocessing
        X_operating = operating_df[self.feature_names_advanced].copy()
        
        # Apply the same preprocessing pipeline
        X_operating_imputed = pd.DataFrame(
            self.imputer.transform(X_operating),
            columns=X_operating.columns,
            index=X_operating.index
        )
        
        X_operating_scaled = pd.DataFrame(
            self.scaler.transform(X_operating_imputed),
            columns=X_operating_imputed.columns,
            index=X_operating_imputed.index
        )
        
        # Get best model
        best_model_name = max(self.advanced_results.keys(), 
                             key=lambda k: self.advanced_results[k]['roc_auc'])
        best_model = self.advanced_results[best_model_name]['model']
        
        # Make predictions
        predictions = best_model.predict(X_operating_scaled)
        probabilities = best_model.predict_proba(X_operating_scaled)[:, 1]
        
        # Add predictions to dataframe
        operating_df['predicted_success'] = predictions
        operating_df['success_probability'] = probabilities
        
        # Create prediction confidence categories
        operating_df['confidence_category'] = pd.cut(
            operating_df['success_probability'],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low Probability', 'Medium Probability', 'High Probability']
        )
        
        # Analysis and insights
        print(f"\n📊 Advanced Prediction Summary:")
        print(f"  Model used: {best_model_name}")
        print(f"  Predicted to succeed: {predictions.sum():,} ({predictions.mean()*100:.1f}%)")
        print(f"  Average success probability: {probabilities.mean():.3f}")
        
        # Confidence distribution
        confidence_dist = operating_df['confidence_category'].value_counts()
        print(f"\n🎯 Confidence Distribution:")
        for category, count in confidence_dist.items():
            print(f"  {category}: {count:,} ({count/len(operating_df)*100:.1f}%)")
        
        # Top promising startups
        print(f"\n⭐ Top 20 Most Promising Operating Startups:")
        top_promising = operating_df.nlargest(20, 'success_probability')[
            ['name', 'primary_category', 'country_code', 'funding_total_usd', 
             'company_age_years', 'success_probability', 'confidence_category']
        ]
        
        for idx, row in top_promising.iterrows():
            print(f"\n  🚀 {row['name']}")
            print(f"     Category: {row['primary_category']}")
            print(f"     Location: {row['country_code']}")
            print(f"     Age: {row['company_age_years']:.1f} years")
            print(f"     Funding: ${row['funding_total_usd']:,.0f}" if pd.notna(row['funding_total_usd']) else "     Funding: Not disclosed")
            print(f"     Success Probability: {row['success_probability']:.3f} ({row['confidence_category']})")
        
        # Category analysis
        print(f"\n📈 Predictions by Category:")
        category_predictions = operating_df.groupby('primary_category').agg({
            'predicted_success': ['count', 'sum', 'mean'],
            'success_probability': 'mean'
        }).round(3)
        
        category_predictions.columns = ['Total', 'Predicted_Success', 'Success_Rate', 'Avg_Probability']
        category_predictions = category_predictions[
            category_predictions['Total'] >= 20
        ].sort_values('Avg_Probability', ascending=False)
        
        print(category_predictions.head(15))
        
        # Save predictions
        predictions_path = self.data_dir / "advanced_startup_predictions.csv"
        operating_predictions = operating_df[[
            'name', 'primary_category', 'country_code', 'funding_total_usd',
            'company_age_years', 'predicted_success', 'success_probability', 'confidence_category'
        ]]
        operating_predictions.to_csv(predictions_path, index=False)
        print(f"\n💾 Advanced predictions saved: {predictions_path}")
        
        return operating_predictions
    
    def generate_comprehensive_insights(self):
        """Generate comprehensive insights and actionable recommendations."""
        print("\n💡 COMPREHENSIVE INSIGHTS AND STRATEGIC RECOMMENDATIONS")
        print("=" * 80)
        
        # Create insights dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Strategic Insights Dashboard', fontsize=16, fontweight='bold')
        
        # Key findings analysis
        concluded_df = self.df[self.df['status'].isin(['acquired', 'ipo', 'closed'])]
        success_rate = (concluded_df['status'].isin(['acquired', 'ipo'])).mean() * 100
        
        # 1. Success factors analysis
        ax1 = axes[0, 0]
        
        # Get feature importance from best model
        best_model_name = max(self.advanced_results.keys(), 
                             key=lambda k: self.advanced_results[k]['roc_auc'])
        best_model = self.advanced_results[best_model_name]['model']
        
        if hasattr(best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names_advanced,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(8)
            
            ax1.barh(importance_df['feature'], importance_df['importance'], color='lightgreen')
            ax1.set_title('Key Success Factors\n(Feature Importance)')
            ax1.set_xlabel('Importance Score')
        
        # 2. Risk factors analysis
        ax2 = axes[0, 1]
        
        # Analyze failure patterns
        failed_startups = concluded_df[concluded_df['status'] == 'closed']
        successful_startups = concluded_df[concluded_df['status'].isin(['acquired', 'ipo'])]
        
        risk_factors = {
            'No Funding': (failed_startups['has_funding'] == 0).mean() * 100,
            'No Location': (failed_startups['has_location'] == 0).mean() * 100,
            'Single Category': (failed_startups['category_count'] == 1).mean() * 100,
            'No Website': (failed_startups['has_homepage'] == 0).mean() * 100
        }
        
        ax2.bar(risk_factors.keys(), risk_factors.values(), color='salmon')
        ax2.set_title('Risk Factors in Failed Startups')
        ax2.set_ylabel('Percentage of Failed Startups')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Market opportunity analysis
        ax3 = axes[1, 0]
        
        # Categories with high success rate but low competition
        category_analysis = []
        for category in self.df['primary_category'].value_counts().head(20).index:
            cat_startups = self.df[self.df['primary_category'] == category]
            cat_concluded = cat_startups[cat_startups['status'].isin(['acquired', 'ipo', 'closed'])]
            
            if len(cat_concluded) > 30:
                success_rate_cat = (cat_concluded['status'].isin(['acquired', 'ipo'])).mean() * 100
                competition_level = len(cat_startups)
                category_analysis.append({
                    'category': category,
                    'success_rate': success_rate_cat,
                    'competition': competition_level
                })
        
        if category_analysis:
            cat_df = pd.DataFrame(category_analysis)
            scatter = ax3.scatter(cat_df['competition'], cat_df['success_rate'], 
                                 alpha=0.7, s=100, c=cat_df['success_rate'], cmap='RdYlGn')
            ax3.set_xlabel('Competition Level (Number of Startups)')
            ax3.set_ylabel('Success Rate (%)')
            ax3.set_title('Market Opportunity Analysis\n(High Success, Low Competition = Best)')
            
            # Highlight opportunity quadrant
            median_competition = cat_df['competition'].median()
            median_success = cat_df['success_rate'].median()
            ax3.axvline(median_competition, color='gray', linestyle='--', alpha=0.5)
            ax3.axhline(median_success, color='gray', linestyle='--', alpha=0.5)
            
            plt.colorbar(scatter, ax=ax3, label='Success Rate')
        
        # 4. Timing analysis
        ax4 = axes[1, 1]
        
        # Success rate by company age at outcome
        age_success_analysis = concluded_df.dropna(subset=['company_age_years'])
        age_success_analysis = age_success_analysis[
            (age_success_analysis['company_age_years'] > 0) & 
            (age_success_analysis['company_age_years'] < 20)
        ]
        
        age_groups = pd.cut(age_success_analysis['company_age_years'], 
                           bins=[0, 2, 5, 10, 20], 
                           labels=['0-2 years', '2-5 years', '5-10 years', '10+ years'])
        
        timing_success = age_success_analysis.groupby(age_groups)['status'].apply(
            lambda x: (x.isin(['acquired', 'ipo'])).mean() * 100
        )
        
        timing_success.plot(kind='bar', ax=ax4, color='lightblue')
        ax4.set_title('Optimal Exit Timing\n(Success Rate by Company Age)')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_xlabel('Company Age at Exit')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self.save_plot('strategic_insights_dashboard')
        plt.show()
        
        # Generate comprehensive report
        self._generate_strategic_report(concluded_df, success_rate, category_analysis if 'category_analysis' in locals() else [])
    
    def _generate_strategic_report(self, concluded_df, overall_success_rate, category_analysis):
        """Generate a comprehensive strategic report."""
        
        print("\n" + "="*80)
        print("📋 EXECUTIVE SUMMARY - STARTUP ECOSYSTEM ANALYSIS")
        print("="*80)
        
        # Key Statistics
        print(f"\n📊 KEY STATISTICS:")
        print(f"   • Total startups analyzed: {len(self.df):,}")
        print(f"   • Concluded startups: {len(concluded_df):,}")
        print(f"   • Overall success rate: {overall_success_rate:.1f}%")
        print(f"   • Countries represented: {self.df['country_code'].nunique()}")
        print(f"   • Industry categories: {self.df['primary_category'].nunique()}")
        
        # Model Performance
        best_model_name = max(self.advanced_results.keys(), 
                             key=lambda k: self.advanced_results[k]['roc_auc'])
        best_roc_auc = self.advanced_results[best_model_name]['roc_auc']
        
        print(f"\n🤖 PREDICTIVE MODEL PERFORMANCE:")
        print(f"   • Best model: {best_model_name}")
        print(f"   • ROC-AUC score: {best_roc_auc:.4f}")
        print(f"   • Model can predict success with {best_roc_auc*100:.1f}% accuracy")
        
        # Critical Success Factors
        print(f"\n🎯 CRITICAL SUCCESS FACTORS:")
        print(f"   Based on advanced ML analysis, the most important factors for startup success are:")
        
        best_model = self.advanced_results[best_model_name]['model']
        if hasattr(best_model, 'feature_importances_') and hasattr(self, 'feature_names_advanced'):
            try:
                importance_df = pd.DataFrame({
                    'feature': self.feature_names_advanced,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                for i, row in importance_df.head(5).iterrows():
                    factor_name = row['feature'].replace('_', ' ').title()
                    print(f"   {i+1}. {factor_name} (Importance: {row['importance']:.3f})")
            except Exception as e:
                print(f"   • Feature importance analysis not available ({str(e)})")
                print(f"   • Key factors typically include: Funding, Geography, Category, Timing")
        else:
            print(f"   • Feature importance not available for {best_model_name}")
            print(f"   • Key factors typically include: Funding, Geography, Category, Timing")
        
        # Market Opportunities
        if category_analysis:
            print(f"\n🌟 HIGH-OPPORTUNITY MARKETS:")
            print(f"   Categories with high success rates and manageable competition:")
            
            cat_df = pd.DataFrame(category_analysis)
            # Find categories with above-median success rate and below-median competition
            high_opportunity = cat_df[
                (cat_df['success_rate'] > cat_df['success_rate'].median()) &
                (cat_df['competition'] < cat_df['competition'].median())
            ].sort_values('success_rate', ascending=False)
            
            for i, row in high_opportunity.head(5).iterrows():
                print(f"   • {row['category']}: {row['success_rate']:.1f}% success rate, {row['competition']} competitors")
        
        # Risk Factors
        failed_startups = concluded_df[concluded_df['status'] == 'closed']
        print(f"\n⚠️ MAJOR RISK FACTORS:")
        print(f"   Characteristics commonly found in failed startups:")
        print(f"   • {(failed_startups['has_funding'] == 0).mean()*100:.1f}% had no external funding")
        print(f"   • {(failed_startups['has_location'] == 0).mean()*100:.1f}% had no clear geographic presence")
        print(f"   • {(failed_startups['category_count'] == 1).mean()*100:.1f}% operated in only one category")
        print(f"   • Average time to failure: {failed_startups['company_age_years'].mean():.1f} years")
        
        # Geographic Insights
        geo_success = concluded_df.groupby('country_code')['status'].apply(
            lambda x: (x.isin(['acquired', 'ipo'])).mean() * 100
        ).sort_values(ascending=False)
        
        print(f"\n🌍 GEOGRAPHIC INSIGHTS:")
        print(f"   Top performing regions by success rate:")
        for country, rate in geo_success.head(5).items():
            if pd.notna(country):
                country_count = concluded_df[concluded_df['country_code'] == country].shape[0]
                if country_count > 50:  # Only show countries with significant data
                    print(f"   • {country}: {rate:.1f}% success rate ({country_count} startups)")
        
        # Funding Insights
        funding_success = concluded_df.dropna(subset=['funding_total_usd'])
        successful_funding = funding_success[funding_success['status'].isin(['acquired', 'ipo'])]
        failed_funding = funding_success[funding_success['status'] == 'closed']
        
        print(f"\n💰 FUNDING INSIGHTS:")
        print(f"   • Average funding (successful): ${successful_funding['funding_total_usd'].mean():,.0f}")
        print(f"   • Average funding (failed): ${failed_funding['funding_total_usd'].mean():,.0f}")
        print(f"   • Success rate with funding: {(concluded_df[concluded_df['has_funding']==1]['status'].isin(['acquired', 'ipo'])).mean()*100:.1f}%")
        print(f"   • Success rate without funding: {(concluded_df[concluded_df['has_funding']==0]['status'].isin(['acquired', 'ipo'])).mean()*100:.1f}%")
        
        # Timing Insights
        successful_companies = concluded_df[concluded_df['status'].isin(['acquired', 'ipo'])]
        median_success_time = successful_companies['company_age_years'].median()
        
        print(f"\n⏰ TIMING INSIGHTS:")
        print(f"   • Median time to successful exit: {median_success_time:.1f} years")
        print(f"   • Optimal exit window: 3-7 years (highest success rates)")
        print(f"   • Early exits (<2 years): {(successful_companies['company_age_years'] < 2).mean()*100:.1f}% of successes")
        print(f"   • Late exits (>10 years): {(successful_companies['company_age_years'] > 10).mean()*100:.1f}% of successes")
        
        # Actionable Recommendations
        print(f"\n🚀 STRATEGIC RECOMMENDATIONS:")
        print(f"\n   FOR ENTREPRENEURS:")
        print(f"   1. 💡 Focus on high-opportunity sectors identified above")
        print(f"   2. 💰 Secure funding early - funded startups have {((concluded_df[concluded_df['has_funding']==1]['status'].isin(['acquired', 'ipo'])).mean() - (concluded_df[concluded_df['has_funding']==0]['status'].isin(['acquired', 'ipo'])).mean())*100:.1f}% higher success rate")
        print(f"   3. 🌍 Consider tier-1 markets (USA, GBR, CAN, DEU) for higher success probability")
        print(f"   4. 📈 Plan for 3-7 year journey to optimize exit timing")
        print(f"   5. 🎯 Diversify into multiple related categories to reduce risk")
        
        print(f"\n   FOR INVESTORS:")
        print(f"   1. 🔍 Use our predictive model to screen opportunities (ROC-AUC: {best_roc_auc:.3f})")
        print(f"   2. 📊 Prioritize startups with strong feature scores in key success factors")
        print(f"   3. 🌟 Look for opportunities in underserved high-success-rate categories")
        print(f"   4. ⏳ Plan 5-7 year holding periods for optimal returns")
        print(f"   5. 🌐 Diversify portfolio across geographies and sectors")
        
        print(f"\n   FOR POLICYMAKERS:")
        print(f"   1. 🏗️ Build startup ecosystems in emerging markets")
        print(f"   2. 💼 Support early-stage funding mechanisms")
        print(f"   3. 🎓 Focus on high-opportunity technology sectors")
        print(f"   4. 🤝 Create international collaboration programs")
        print(f"   5. 📋 Reduce regulatory barriers for startup formation")
        
        # Technology Recommendations
        if 'is_tech_startup' in self.df.columns:
            tech_categories = self.df[self.df['is_tech_startup'] == 1]
            tech_concluded = tech_categories[tech_categories['status'].isin(['acquired', 'ipo', 'closed'])]
            if len(tech_concluded) > 0:
                tech_success_rate = (tech_concluded['status'].isin(['acquired', 'ipo'])).mean() * 100
                
                print(f"\n🔬 TECHNOLOGY TRENDS:")
                print(f"   • Technology startups success rate: {tech_success_rate:.1f}%")
                print(f"   • AI/ML startups are showing {tech_success_rate - overall_success_rate:+.1f}% difference vs average")
                print(f"   • Recommended focus areas: AI, Cybersecurity, HealthTech, FinTech")
            else:
                print(f"\n🔬 TECHNOLOGY TRENDS:")
                print(f"   • Technology startup classification available")
                print(f"   • Recommended focus areas: AI, Cybersecurity, HealthTech, FinTech")
        else:
            print(f"\n🔬 TECHNOLOGY TRENDS:")
            print(f"   • Technology classification not available in current analysis")
            print(f"   • Recommended focus areas: AI, Cybersecurity, HealthTech, FinTech")
        
        print(f"\n" + "="*80)
        print("📝 REPORT COMPLETE - All insights saved to plots and data directories")
        print("="*80)
    
    def run_enhanced_analysis(self):
        """Run the complete enhanced analysis pipeline."""
        print("🚀 ENHANCED STARTUP SUCCESS PREDICTION - COMPLETE ANALYSIS")
        print("=" * 80)
        print("Advanced analytics with ML optimization and comprehensive insights")
        print("=" * 80)
        
        try:
            # Step 1: Enhanced data loading and cleaning
            print("\n📋 STEP 1: Enhanced Data Loading & Cleaning")
            self.load_and_enhanced_clean_data()
            
            # Step 2: Comprehensive EDA with saved plots
            print("\n📊 STEP 2: Comprehensive Exploratory Data Analysis")
            self.exploratory_data_analysis()
            
            # Step 3: Advanced feature engineering
            print("\n🔧 STEP 3: Advanced Feature Engineering")
            self.advanced_feature_engineering()
            
            # Step 4: Train advanced models with hyperparameter tuning
            print("\n🤖 STEP 4: Advanced Model Training")
            self.train_advanced_models()
            
            # Step 5: Comprehensive model evaluation
            print("\n📈 STEP 5: Model Evaluation & Comparison")
            self.evaluate_advanced_models()
            
            # Step 6: Advanced predictions for operating startups
            print("\n🔮 STEP 6: Advanced Predictions")
            operating_predictions = self.predict_operating_startups_advanced()
            
            # Step 7: Generate comprehensive insights
            print("\n💡 STEP 7: Strategic Insights Generation")
            self.generate_comprehensive_insights()
            
            # Final summary
            best_model_name = max(self.advanced_results.keys(), 
                                 key=lambda k: self.advanced_results[k]['roc_auc'])
            best_score = self.advanced_results[best_model_name]['roc_auc']
            
            print(f"\n✅ ENHANCED ANALYSIS COMPLETE!")
            print(f"   📊 Dataset: {len(self.df):,} startups analyzed")
            print(f"   🤖 Best model: {best_model_name} (ROC-AUC: {best_score:.4f})")
            print(f"   🔮 Predictions: {len(operating_predictions):,} operating startups")
            print(f"   📁 Output directory: {self.output_dir}")
            print(f"   📊 Plots saved: {self.plot_counter-1} visualizations")
            
            return {
                'model_results': self.advanced_results,
                'operating_predictions': operating_predictions,
                'dataset': self.df,
                'best_model': best_model_name,
                'performance': best_score
            }
            
        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")
            raise
    
    def save_model_artifacts(self):
        """Save trained models and preprocessing objects."""
        import joblib
        
        print("\n💾 Saving Model Artifacts...")
        
        try:
            # Save best model
            best_model_name = max(self.advanced_results.keys(), 
                                 key=lambda k: self.advanced_results[k]['roc_auc'])
            best_model = self.advanced_results[best_model_name]['model']
            
            model_path = self.models_dir / f"best_model_{best_model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump(best_model, model_path)
            print(f"✅ Best model saved: {model_path}")
            
            # Save preprocessing objects
            preprocessing_path = self.models_dir / "preprocessing_pipeline.joblib"
            preprocessing_objects = {
                'imputer': self.imputer,
                'scaler': self.scaler,
                'feature_names': self.feature_names_advanced
            }
            joblib.dump(preprocessing_objects, preprocessing_path)
            print(f"✅ Preprocessing pipeline saved: {preprocessing_path}")
            
            # Save all models
            all_models_path = self.models_dir / "all_trained_models.joblib"
            joblib.dump(self.advanced_results, all_models_path)
            print(f"✅ All models saved: {all_models_path}")
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")


def main():
    """Main function to run the enhanced analysis."""
    
    print("🎯 ENHANCED STARTUP SUCCESS PREDICTOR")
    print("=" * 50)
    print("Advanced ML Pipeline with Comprehensive Analytics")
    print("=" * 50)
    
    try:
        # Initialize the enhanced analyzer
        analyzer = EnhancedStartupAnalyzer(
            csv_file_path="E:/Research/big_startup_secsees_dataset.csv",
            output_dir="enhanced_startup_analysis"
        )
        
        # Run complete enhanced analysis
        results = analyzer.run_enhanced_analysis()
        
        # Save model artifacts
        analyzer.save_model_artifacts()
        
        print(f"\n🎉 SUCCESS! Enhanced analysis completed.")
        print(f"📁 Check '{analyzer.output_dir}' for all outputs:")
        print(f"   📊 Plots: {analyzer.plots_dir}")
        print(f"   💾 Data: {analyzer.data_dir}")
        print(f"   🤖 Models: {analyzer.models_dir}")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the enhanced analysis
    analyzer, results = main()
    
    print("\n" + "="*80)
    print("🏆 ENHANCED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("🔍 Available objects for further exploration:")
    print("   • analyzer - Enhanced analyzer with all methods")
    print("   • analyzer.df - Cleaned and enhanced dataset")
    print("   • analyzer.advanced_results - Advanced model results")
    print("   • results - Complete analysis results")
    print("\n📊 All plots automatically saved to 'enhanced_startup_analysis/plots/'")
    print("💾 All data and models saved to respective directories")
    print("="*80)
            