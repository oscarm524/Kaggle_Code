"""
Feature engineering module for customer churn prediction.

This module provides functions to create various engineered features
from customer data to improve churn prediction models.
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1. Service Engagement Features
# =============================================================================

def count_services(row, service_cols):
    """
    Count the number of active services for a customer.
    
    Parameters
    ----------
    row : pd.Series
        A row from the dataframe
    service_cols : list
        List of service column names
        
    Returns
    -------
    int
        Count of active services
    """
    count = 0
    no_service_values = [
        'No', 'No internet service', 'No phone service', None
    ]
    for col in service_cols:
        value = row.get(col)
        if value not in no_service_values:
            count += 1
    return count


def add_service_engagement_features(df):
    """
    Add service engagement related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new service engagement features
    """
    df = df.copy()
    
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Total services count
    df['TotalServices'] = df.apply(
        lambda row: count_services(row, service_cols), axis=1
    )
    
    # Service density (services per month)
    df['ServiceDensity'] = df['TotalServices'] / (df['tenure'] + 1)
    
    # Premium services count
    premium_services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport'
    ]
    df['PremiumServicesCount'] = sum([
        (df[col] == "Yes").astype(int)
        for col in premium_services if col in df.columns
    ])
    
    # Entertainment services
    entertainment = ['StreamingTV', 'StreamingMovies']
    df['EntertainmentServicesCount'] = sum([
        (df[col] == "Yes").astype(int)
        for col in entertainment if col in df.columns
    ])
    
    # Has any protection services
    df['HasProtection'] = (
        (df.get('OnlineSecurity') == "Yes") |
        (df.get('OnlineBackup') == "Yes") |
        (df.get('DeviceProtection') == "Yes") |
        (df.get('TechSupport') == "Yes")
    ).astype(int)
    
    # Service adoption rate
    total_possible_services = len(service_cols)
    df['ServiceAdoptionRate'] = df['TotalServices'] / total_possible_services
    
    return df


# =============================================================================
# 2. Interaction Features
# =============================================================================

def add_interaction_features(df):
    """
    Add interaction features between categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new interaction features
    """
    df = df.copy()
    
    # Payment behavior interactions
    df['PaperlessBilling_PaymentMethod'] = (
        df['PaperlessBilling'].astype(str) + "_" +
        df['PaymentMethod'].astype(str)
    )
    
    # Contract + Internet Service (loyalty vs service type)
    df['Contract_InternetService'] = (
        df['Contract'].astype(str) + "_" +
        df['InternetService'].astype(str)
    )
    
    # Senior + Partner + Dependents (household structure)
    df['SeniorCitizen_Partner_Dependents'] = (
        df['SeniorCitizen'].astype(str) + "_" +
        df['Partner'].astype(str) + "_" +
        df['Dependents'].astype(str)
    )
    
    return df


# =============================================================================
# 3. Financial Features
# =============================================================================

def add_financial_features(df):
    """
    Add financial and pricing related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new financial features
    """
    df = df.copy()
    
    # Average monthly charge (total / tenure)
    df['AvgMonthlyCharge'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # Charge difference (actual vs expected)
    df['ChargeDifference'] = (
        df['TotalCharges'] - (df['MonthlyCharges'] * df['tenure'])
    )
    
    # Price per service
    df['PricePerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)
    
    # Binned monthly charges
    df['MonthlyCharges_Binned'] = pd.cut(
        df['MonthlyCharges'],
        bins=[0, 30, 60, 90, float('inf')],
        labels=['Low', 'Medium', 'High', 'VeryHigh']
    )
    
    # High value customer (top quartile)
    df['IsHighValueCustomer'] = (
        df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)
    ).astype(int)
    
    # Value ratio (what they pay vs tenure)
    df['ValueRatio'] = (
        df['TotalCharges'] / (df['tenure'] + 1) / (df['TotalServices'] + 1)
    )
    
    return df


# =============================================================================
# 4. Tenure-based Features
# =============================================================================

def add_tenure_features(df):
    """
    Add tenure related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new tenure features
    """
    df = df.copy()
    
    # Tenure categories
    df['TenureGroup'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, float('inf')],
        labels=['0-1yr', '1-2yr', '2-4yr', '4yr+']
    )
    
    # New customer flag
    df['IsNewCustomer'] = (df['tenure'] <= 6).astype(int)
    
    # Long-term customer
    df['IsLongTermCustomer'] = (df['tenure'] >= 48).astype(int)
    
    # Tenure squared (non-linear relationship)
    df['TenureSquared'] = df['tenure'] ** 2
    
    # Tenure log (diminishing returns)
    df['TenureLog'] = np.log1p(df['tenure'])
    
    # Contract commitment score (longer contract = higher score)
    contract_map = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    df['ContractMonths'] = df['Contract'].map(contract_map)
    
    return df


# =============================================================================
# 5. Service Complexity Features
# =============================================================================

def add_service_complexity_features(df):
    """
    Add service complexity and type related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new service complexity features
    """
    df = df.copy()
    
    # Has phone but no internet
    df['PhoneOnly'] = (
        (df['PhoneService'] == "Yes") &
        (df['InternetService'] == "No")
    ).astype(int)
    
    # Has internet but no phone
    df['InternetOnly'] = (
        (df['PhoneService'] == "No") &
        (df['InternetService'] != "No")
    ).astype(int)
    
    # Fiber optic customer (typically higher cost)
    df['IsFiberOptic'] = (
        df['InternetService'] == "Fiber optic"
    ).astype(int)
    
    # Has multiple lines
    df['HasMultipleLines'] = (df['MultipleLines'] == "Yes").astype(int)
    
    return df


# =============================================================================
# 6. Risk Indicators
# =============================================================================

def add_risk_indicator_features(df):
    """
    Add churn risk indicator features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
        
    Returns
    -------
    pd.DataFrame
        Dataframe with new risk indicator features
    """
    df = df.copy()
    
    # Month-to-month with high charges (high churn risk)
    df['MonthToMonth_HighCharge'] = (
        (df['Contract'] == "Month-to-month") &
        (df['MonthlyCharges'] > df['MonthlyCharges'].median())
    ).astype(int)
    
    # Electronic check payment (often associated with higher churn)
    df['ElectronicCheck'] = (
        df['PaymentMethod'] == "Electronic check"
    ).astype(int)
    
    # Single person with no protection
    df['SingleNoProtection'] = (
        (df['Partner'] == "No") &
        (df['Dependents'] == "No") 
    ).astype(int)
    
    return df


# =============================================================================
# 7. Master Function
# =============================================================================

def engineer_features(df, feature_groups='all'):
    """
    Apply all feature engineering transformations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_groups : str or list, default='all'
        Which feature groups to apply. Options:
        - 'all': Apply all feature engineering
        - list: Specify groups ['service', 'interaction', 'financial',
                'tenure', 'complexity', 'risk']
        
    Returns
    -------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    if feature_groups == 'all':
        feature_groups = [
            'service', 'interaction', 'financial',
            'tenure', 'complexity', 'risk'
        ]
    
    if 'service' in feature_groups:
        df = add_service_engagement_features(df)
    
    if 'interaction' in feature_groups:
        df = add_interaction_features(df)
    
    if 'financial' in feature_groups:
        df = add_financial_features(df)
    
    if 'tenure' in feature_groups:
        df = add_tenure_features(df)
    
    if 'complexity' in feature_groups:
        df = add_service_complexity_features(df)
    
    if 'risk' in feature_groups:
        df = add_risk_indicator_features(df)
    
    return df