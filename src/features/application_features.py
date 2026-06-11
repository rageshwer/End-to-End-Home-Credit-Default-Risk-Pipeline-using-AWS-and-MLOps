'''This script performs feature engineering on application (train and test) datasets and returns the processed datasets.'''
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_application_features(app:pd.DataFrame)->pd.DataFrame:
    '''This function performs feature engineering on the application dataset.'''

    # Replacing the value 365243 in the 'DAYS_EMPLOYED' column with NaN, as it is likely a placeholder for missing data.
    app.loc[app['DAYS_EMPLOYED']==365243,'DAYS_EMPLOYED']=np.nan
    
    # Converting days into years
    app['age_years'] = abs(app['DAYS_BIRTH']) / 365
    app['employment_years'] = abs(app['DAYS_EMPLOYED']) / 365
    
    #========================================================================================
    # Creating features from income, credit, and annuity
    #========================================================================================
    eps=1e-6

    # EMI/NMI ratio
    app['annuity_to_income']=app['AMT_ANNUITY']/(app['AMT_INCOME_TOTAL']+eps)

    # credit to income ratio
    app['credit_to_income']=app['AMT_CREDIT']/(app['AMT_INCOME_TOTAL']+eps)

    # goods price to income ratio : weather he can afford the product or not
    app['income_to_price']=app['AMT_INCOME_TOTAL']/(app['AMT_GOODS_PRICE']+eps)
    
    # income and days employed reltion
    app['income_to_days']=app['AMT_INCOME_TOTAL']/((abs(app['DAYS_EMPLOYED'])/365)+eps)
    
    # income and age relation
    app['income_to_age']=app['AMT_INCOME_TOTAL']/((abs(app['DAYS_BIRTH'])/365)+eps)
    
    # credit to annuity ratio
    app['credit_to_annuity'] =app['AMT_CREDIT'] /(app['AMT_ANNUITY']+eps)
    
    # goods to credit
    app['goods_to_credit'] = app['AMT_GOODS_PRICE'] /(app['AMT_CREDIT']+eps)
    
    # income divided into family members
    app['income_per_person'] = app['AMT_INCOME_TOTAL']/(app['CNT_FAM_MEMBERS']+eps)
    
    # job to age ratio
    app['employment_age_ratio'] = app['DAYS_EMPLOYED']/(app['DAYS_BIRTH']+eps)
    
    # ========================================================================================
    # Family related features
    # ========================================================================================

    # child ratio in family
    app['child_ratio'] = app['CNT_CHILDREN']/(app['CNT_FAM_MEMBERS'] + eps)
    
    # adult count in family
    app['adult_count'] = app['CNT_FAM_MEMBERS'] -app['CNT_CHILDREN']
    
    # credit per person
    app['credit_per_person'] = app['AMT_CREDIT'] /(app['CNT_FAM_MEMBERS'] + eps)
    
    # annuity per person
    app['annuity_per_person'] = app['AMT_ANNUITY'] /(app['CNT_FAM_MEMBERS'] + eps)
    

    #========================================================================================
    # External source related features
    #========================================================================================

    ext_cols=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

    # mean
    app['ext_mean']=app[ext_cols].mean(axis=1)

    # median
    
    app['ext_med']=app[ext_cols].median(axis=1)

    # max
    app['ext_max']=app[ext_cols].max(axis=1)

    # min
    app['ext_min']=app[ext_cols].min(axis=1)

    # std
    app['ext_std']=app[ext_cols].std(axis=1)

    # product
    app['ext_mul']=app['EXT_SOURCE_1']*app['EXT_SOURCE_2']*app['EXT_SOURCE_3']

    # range
    app['ext_range']=app['ext_max']-app['ext_min']

    # missing
    app['ext_missing']=app[ext_cols].isna().sum(axis=1)

    # interactions
    app['ext_mean_x_age'] = (
            app['ext_mean']
            * (abs(app['DAYS_BIRTH']) / 365)
        )
    
    app['ext_mean_x_employment'] = (
            app['ext_mean']
            * (abs(app['DAYS_EMPLOYED']) / 365)
        )

    app['ext_mean_x_income'] = (
            app['ext_mean']
            * app['AMT_INCOME_TOTAL']
        )

    #========================================================================================
    # Documents provided flags
    #========================================================================================
    doc_cols=[col for col in app.columns if 'FLAG_DOCUMENT_' in col]
    app['total_docs_provided']=app[doc_cols].sum(axis=1)
    app['docs_absent']=len(doc_cols)-app['total_docs_provided']

    #========================================================================================
    # Contact
    #========================================================================================
    contact_cols = [
    'FLAG_MOBIL',
    'FLAG_EMP_PHONE',
    'FLAG_WORK_PHONE',
    'FLAG_CONT_MOBILE',
    'FLAG_PHONE',
    'FLAG_EMAIL'
    ]

    app['contact_count'] = app[contact_cols].sum(axis=1)
    app['missing_contact_count'] = (
        len(contact_cols) -
        app[contact_cols].sum(axis=1)
    )

    #========================================================================================
    # Address columns
    #========================================================================================
    addr_cols = [
    'REG_REGION_NOT_LIVE_REGION',
    'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION',
    'REG_CITY_NOT_LIVE_CITY',
    'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY'
    ]
    # address mismatch
    app['address_mismatch_count'] = (
        app[addr_cols]
        .sum(axis=1)
    )

    #========================================================================================
    # Registration and ID columns
    #========================================================================================

    app['registration_age_ratio'] = (
    abs(app['DAYS_REGISTRATION'])
        /abs(app['DAYS_BIRTH'])
        )
    
    app['id_publish_age_ratio'] = (
        abs(app['DAYS_ID_PUBLISH'])
        /
        abs(app['DAYS_BIRTH'])
    )

    #========================================================================================
    # House columns
    #========================================================================================
    house_cols = [c for c in app.columns if '_AVG' in c or '_MODE' in c or '_MEDI' in c]
    app['housing_missing_count'] = app[house_cols].isnull().sum(axis=1)


    #========================================================================================
    # Categorical features
    #========================================================================================
    # binary encoding
    app['CODE_GENDER_BIN'] = (
        app['CODE_GENDER']
        .map({'M': 0, 'F': 1})
        .fillna(2)
        .astype('int8')
    )

    app['FLAG_OWN_CAR_BIN'] = (
        app['FLAG_OWN_CAR']
        .map({'N': 0, 'Y': 1})
        .astype('int8')
    )

    app['FLAG_OWN_REALTY_BIN'] = (
        app['FLAG_OWN_REALTY']
        .map({'N': 0, 'Y': 1})
        .astype('int8')
    )

    # missing as category
    cat_cols = app.select_dtypes(include='object').columns
    for col in cat_cols:
        app[col] = app[col].fillna('Missing')

    # frequency columns
    freq_cols = [
    'ORGANIZATION_TYPE',
    'OCCUPATION_TYPE',
    'NAME_INCOME_TYPE'
    ]

    for col in freq_cols:
        freq_map = (
            app[col]
            .value_counts(normalize=True)
            .to_dict()
        )
        app[f'{col}_FREQ'] = app[col].map(freq_map).fillna(0)

    # organization is rare or not
    org_freq = app['ORGANIZATION_TYPE'].value_counts(normalize=True)
    rare_orgs = org_freq[org_freq < 0.01].index
    app['ORG_IS_RARE'] = app['ORGANIZATION_TYPE'].isin(rare_orgs).astype('int8')

    # self employed
    app['SELF_EMPLOYED_FLAG'] = (
        (app['NAME_INCOME_TYPE'] == 'Self-employed')
        .astype('int8')
    )

    # employment segment
    employment_map = {

        'Working': 'Working',
        'Commercial associate': 'Working',
        'State servant': 'Working',

        'Pensioner': 'Pensioner',

        'Student': 'Other',
        'Unemployed': 'Other',
        'Businessman': 'Other',
        'Maternity leave': 'Other'
    }

    app['EMPLOYMENT_SEGMENT'] = (
        app['NAME_INCOME_TYPE']
        .map(employment_map)
        .fillna('Other')
    )

    # skill group
    high_skill = [
        'Managers',
        'Accountants',
        'Core staff',
        'HR staff',
        'IT staff',
        'Medicine staff',
        'High skill tech staff'
    ]

    app['OCCUPATION_SKILL_GROUP'] = np.where(
        app['OCCUPATION_TYPE'].isin(high_skill),
        'HighSkill',
        'Other'
    )

    # categories marking for LGBM
    lightgbm_cat_cols = [

        'NAME_CONTRACT_TYPE',
        'NAME_TYPE_SUITE',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE',
        'OCCUPATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START',
        'ORGANIZATION_TYPE',
        'FONDKAPREMONT_MODE',
        'HOUSETYPE_MODE',
        'WALLSMATERIAL_MODE',
        'EMERGENCYSTATE_MODE',

        'EMPLOYMENT_SEGMENT',
        'OCCUPATION_SKILL_GROUP'
    ]

    for col in lightgbm_cat_cols:
        app[col] = app[col].astype('category')


    
    return app