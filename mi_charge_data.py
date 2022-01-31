
import pandas as pd
import numpy as np
import fastparquet

def get_data(hcpcs=True):
    df = load_data()
    
    if hcpcs:
        df = vectorize_hcpcs(df)
    else: 
        df = df.drop(columns=['HCPCS_Cd'])
        
    X_train, X_test, y_train, y_test, features = split_data(df)
    X_train, X_test = normalize_data(X_train, X_test)
    
    if hcpcs:
        X_train, X_test = pca_data(X_train, X_test)
        
    return X_train, X_test, y_train, y_test, features


def load_data():

    df = pd.read_parquet('prov_svc_2019_mi.parquet', 'fastparquet')

    # missing value for Rndrng_Prvdr_Crdntls and Rndrng_Prvdr_Gndr
    df = df.fillna('')

    df['Tot_Benes'] = pd.to_numeric(df['Tot_Benes'])
    df['Tot_Srvcs'] = pd.to_numeric(df['Tot_Srvcs'])
    df['Tot_Bene_Day_Srvcs'] = pd.to_numeric(df['Tot_Bene_Day_Srvcs'])
    df['Avg_Sbmtd_Chrg'] = pd.to_numeric(df['Avg_Sbmtd_Chrg'])
    df['Avg_Mdcr_Alowd_Amt'] = pd.to_numeric(df['Avg_Mdcr_Alowd_Amt'])
    df['Avg_Mdcr_Pymt_Amt'] = pd.to_numeric(df['Avg_Mdcr_Pymt_Amt'])
    df['Avg_Mdcr_Stdzd_Amt'] = pd.to_numeric(df['Avg_Mdcr_Stdzd_Amt'])

    # feature engineering: turn category/indicator data to numberic
    df['gender'] = np.where((df.Rndrng_Prvdr_Gndr == 'M'), 1, 0) 
    df['entity'] = np.where(df.Rndrng_Prvdr_Ent_Cd == 'O', 1, 0) # organization or individual
    df['Rndrng_Prvdr_Mdcr_Prtcptg_Ind'] = np.where(df.Rndrng_Prvdr_Mdcr_Prtcptg_Ind == 'Y', 1, 0)
    df['HCPCS_Drug_Ind'] = np.where(df.HCPCS_Drug_Ind == 'Y', 1, 0)
    df['facility'] = np.where(df.Place_Of_Srvc == 'F', 1, 0)

    # provider credentials
    df.Rndrng_Prvdr_Crdntls = df.Rndrng_Prvdr_Crdntls.apply(lambda x: crdntls_map(x))
    df['md'] = df.Rndrng_Prvdr_Crdntls.apply(lambda x: isMD(x))

    # df.Rndrng_Prvdr_RUCA.unique()
    # df.groupby('Rndrng_Prvdr_RUCA').count()
    df['Rndrng_Prvdr_RUCA'] = pd.to_numeric(df['Rndrng_Prvdr_RUCA'])
    df['metro'] = np.where((df.Rndrng_Prvdr_RUCA <=6), 1, 0)
    # df.groupby('metro').count()

    df = df.drop(columns=['Rndrng_NPI', 'Rndrng_Prvdr_Last_Org_Name', 'Rndrng_Prvdr_First_Name', 'Rndrng_Prvdr_MI',                 'Rndrng_Prvdr_St1', 'Rndrng_Prvdr_St2', 'Rndrng_Prvdr_City', 'Rndrng_Prvdr_State_Abrvtn',                 'Rndrng_Prvdr_State_FIPS', 'Rndrng_Prvdr_Zip5', 'Rndrng_Prvdr_RUCA_Desc', 'Rndrng_Prvdr_Cntry',                 'Rndrng_Prvdr_Type', 'HCPCS_Desc'])

    # following columns converted to new binary column
    df = df.drop(columns=['Rndrng_Prvdr_Crdntls', 'Rndrng_Prvdr_Gndr', 'Rndrng_Prvdr_Ent_Cd', 'Rndrng_Prvdr_RUCA', 'Place_Of_Srvc'])

    # drop features not useful after initial evaluation
    df = df.drop(columns=['Rndrng_Prvdr_Mdcr_Prtcptg_Ind', 'HCPCS_Drug_Ind', 'metro'])

    return df


def crdntls_map(crdntls):
    result = crdntls
    if (type(crdntls) == type('str')):        
        result = crdntls.replace ('.','').replace('>','').replace('DOCTOR OF AUDIOLOGY','AUD').replace('DR AUDIOLOGY','AUD').replace('AUDIOLOGIST','AUD').replace('DC CHIROPRACTOR','DC').replace('DOCTOR OF CHIROPRACT','DC').replace('CHIROPRACTOR DC','DC').replace('DC DR OF CHIROPRACTI','DC').replace('DC DOCTOR OF CHIROPR','DC').replace('CHIROPRACTIC','DC')                          .replace('D O','DO').replace('D.O.','DO').replace('D.O','DO').replace('D P M','DPM').replace ('M D','MD').replace ('M.D.','MD').replace ('M. D.','MD').replace('M.D','MD').replace('M>D>','MD').replace('M D','MD').replace('M,D,','MD').replace('N P','NP')                  .replace('NURSE PRACTITIONER','NP').replace('NURSE PRACITIONER','NP').replace('NP NURSE PRACTITIONE','NP').replace('NURSE PRACTIONER','NP').replace('NP- NURSE PRACTITION', 'NP').replace('NURSE PRACTTIONER','NP').replace('NP(WOMEN\'S HEALTH)','NP').replace('FAMILY NURSE PRACTIT','NP').replace('FAMILY NURSE PRACTIO', 'NP').replace('NATALIE BROWN','').replace('ISLAM GOMAA','').replace('FAMILY PRACTICE','').replace('INTERNAL MEDICINE','').replace('OD DR OF OPTOMETRY','OD').replace('LINDSEY WYNKOOP, OD','OD').replace('PA-C','PAC').replace('(PA-C)','PAC').replace('(PAC)','PAC').replace('PA -C','PAC').replace('PHD','PHD').replace('PH.D.','PHD').replace('DOCTORATE IN PHYSICA','PHD') .replace('PHYSICIAN ASSISTANT','PA').replace('PHYSICIANS ASSISTANT','PA').replace('WILLIAM BLAKESLEE PA','PA').replace('P A','PA').replace('PHYSICAL THERAPIST', 'PT').replace('P T','PT').replace('REGISTERED DIETITIAN','RD').replace('SOCIAL WORKER','SW')
    else:
        result = ''
    
    return result

def isMD(crdntls):
    result = 0
    if (('MD' in crdntls) or ('DO' in crdntls) or ('MBBS' in crdntls)):
        result = 1
    return result


def vectorize_hcpcs(df):

    # Get dummies
    df_hcpcs = pd.get_dummies(df, prefix_sep='_', drop_first=False)
    df_hcpcs.head()
    assert len(df.columns) + len(df.HCPCS_Cd.unique()) -1 == len(df_hcpcs.columns)

    # from sklearn.feature_extraction import DictVectorizer
    # # turn df into dict, each row as key-value pairs
    # df_dict = df.to_dict('records')

    # df_h = pd.DataFrame(DictVectorizer(sparse=False).fit_transform(df_dict))
    # assert len(df.columns) + len(df.HCPCS_Cd.unique()) -1 == len(df_h.columns)
    
    return df_hcpcs

# df.columns: Index(['HCPCS_Cd', 'Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs','Avg_Sbmtd_Chrg', 'Avg_Mdcr_Alowd_Amt',
#         'Avg_Mdcr_Pymt_Amt','Avg_Mdcr_Stdzd_Amt', 'gender', 'entity', 'facility', 'md'], dtype='object')

# df_hcpcs.columns: Index(['Tot_Benes', 'Tot_Srvcs', 'Tot_Bene_Day_Srvcs', 'Avg_Sbmtd_Chrg',
#        'Avg_Mdcr_Alowd_Amt', 'Avg_Mdcr_Pymt_Amt', 'Avg_Mdcr_Stdzd_Amt', 'gender', 'entity', 'facility',
#        ...
#        'HCPCS_Cd_Q5111', 'HCPCS_Cd_Q9950', 'HCPCS_Cd_Q9957', 'HCPCS_Cd_Q9961',
#        'HCPCS_Cd_Q9965', 'HCPCS_Cd_Q9966', 'HCPCS_Cd_Q9967', 'HCPCS_Cd_R0070',
#        'HCPCS_Cd_R0075', 'HCPCS_Cd_V2785'],
#       dtype='object', length=2933)

# df_hcpcs.info(): <class 'pandas.core.frame.DataFrame'>Int64Index: 337351 entries, 280 to 10139999
# Columns: 2933 entries, Tot_Benes to HCPCS_Cd_V2785 dtypes: float64(5), int64(6), uint8(2922) memory usage: 971.0 MB


def split_data(df):
    from sklearn.model_selection import train_test_split
    X=df.drop(['Avg_Sbmtd_Chrg'], axis=1)
    y=df['Avg_Sbmtd_Chrg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, X.columns

def normalize_data(X_train, X_test):
    from sklearn.preprocessing import StandardScaler  #Normalizer
    normalizer = StandardScaler() # Normalizer()
    X_train_norm = normalizer.fit_transform(X_train)
    X_test_norm = normalizer.transform(X_test)
    return X_train_norm, X_test_norm

def pca_data(X_train, X_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 15, svd_solver = 'randomized')    
    # fit = pca.fit(X_train_norm)
    # print(("Explained Variance: %s") % (fit.explained_variance_ratio_))
    # print(fit.components_)
    
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    return X_train_pca, X_test_pca

