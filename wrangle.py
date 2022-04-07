import numpy as np
import pandas as pd

import os

import env


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def get_zillow_data():    
    
    '''This function will acquire data from zillow using env file and rename the columns before saving it as CSV'''

    filename = 'zillow.csv'
    if os.path.exists(filename):
        print('Reading from csv file...')
        return pd.read_csv(filename)
      
    query = '''
        SELECT prop.*,
	    	   pred.logerror,
		       pred.transactiondate,	
    		   cons.typeconstructiondesc,
	    	   air.airconditioningdesc,
		       arch.architecturalstyledesc,
    		   build.buildingclassdesc,
	    	   land.propertylandusedesc,		
		       story.storydesc
    	FROM properties_2017 prop
	    	INNER JOIN(SELECT parcelid, logerror, MAX(transactiondate)transactiondate
		    			FROM predictions_2017
			    		GROUP BY parcelid, logerror) pred
				    USING (parcelid)
    		LEFT JOIN typeconstructiontype cons USING (typeconstructiontypeid)
	    	LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
		    LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
    		LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
	    	LEFT JOIN propertylandusetype land USING (propertylandusetypeid)
		    LEFT JOIN storytype story USING (storytypeid)
	    WHERE prop.latitude IS NOT NULL
		    AND prop.longitude IS NOT NULL
    		AND transactiondate LIKE '2017%%';
        '''
    print('Getting a fresh copy from SQL database...')
    df = pd.read_sql(query, get_db_url('zillow'))
    
	# transfer dataframe into a csv for faster access
	print('Saving to csv...')
    df.to_csv(filename, index=False)
    return df

# Function returns a dataframe with number of rows missing in a column 
# and percentage of the column missing
def nulls_by_col(df):
    '''
    This function will take a dataframe and 
    return a dataframe with null values and percentage by column 
    '''
    num_rows_missing = df.isnull().sum()
    pct_rows_missing = num_rows_missing/num_rows
    df_missing = pd.DataFrame({'number_missing_rows': num_rows_missing, 'percent_rows_missing': pct_rows_missing})
    return df_missing

# Function returns a dataframe with number of rows missing 
# and percentage of rows missing
def nulls_by_row(df):
    '''
    This function will take a dataframe and 
    return a dataframe with null values and percentage by row 
    '''
    miss_df = pd.DataFrame(df.isna().sum(axis =1), columns = ['num_cols_missing']).reset_index().groupby('num_cols_missing')
    miss_df = miss_df.count().reset_index().rename(columns = {'index': 'num_rows' })
   
    miss_df['pct_cols_missing'] = miss_df.num_cols_missing/df.shape[1]
    return miss_df

# This function handles the missing values by removing columns and rows with 50% nulls
def handle_missing_values(df, prop_required_column = .5, prop_required_row = .5):
    '''
    function that will drop rows or columns based on the 
    percent of values that are missing
    '''
    n_required_column = round(df.shape[0] * prop_required_column)
    n_required_row = round(df.shape[1] * prop_required_row)
    df = df.dropna(axis=0, thresh=n_required_row)
    df = df.dropna(axis=1, thresh=n_required_column)
    return df  


# This function will take in a dataframe and the columns to be removed
def remove_columns(df, cols_to_remove):  
	'''
    This function will return a dataframe with indicated columns removed
    '''
    df = df.drop(columns=cols_to_remove)
    return df

# This function will remove some extreme outliers based on observation instead of IQR
def remove_outliers(df):
    '''
    remove outliers in bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, lotsizesquarefeet,
    structuretaxvaluedollarcnt, taxvaluedollarcnt, landtaxvaluedollarcnt, taxamount
    '''

    return df[((df.bathroomcnt <= 7) & (df.bedroomcnt <= 7) &  
               (df.bathroomcnt > 0) & 
               (df.bedroomcnt > 0) & 
               (df.calculatedfinishedsquarefeet > 350) & 
               (df.calculatedfinishedsquarefeet < 6000) & 
               (df.lotsizesquarefeet < 500_000) &
               (df.regionidzip < 100_000) &
               (df.structuretaxvaluedollarcnt < 1_000_000) &
               (df.taxvaluedollarcnt < 2_000_000) &
               (df.landtaxvaluedollarcnt < 1_500_000) &
               (df.taxamount < 30_000)
              )]


def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler(copy=True, feature_range=(0,1))
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test


# This function takes all the small functions and combines them to get a clean zillow dataframe
def wrangle_zillow(df):
    '''
    This function takes in a fresh copy of zillow data and cleans it. 
    After this it will be easy to split
    '''
    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)
        
    # drop columns not needed
    cols_to_remove = ['id', 'calculatedbathnbr', 'finishedsquarefeet12', 'fullbathcnt', 'roomcnt','propertycountylandusecode', 
    'propertylandusetypeid', 'censustractandblock', 'rawcensustractandblock', 'assessmentyear']
    
    df = remove_columns(df, cols_to_remove)
    
    # Columns to look for outliers
    df = remove_outliers(df)
    
    # Add column for counties
    df['county'] = np.where(df.fips == 6037, 'Los_Angeles', np.where(df.fips == 6059, 'Orange', 'Ventura'))    

    # replace nulls with median values for select columns
    lot_median = df.lotsizesquarefeet.median()
    df.lotsizesquarefeet.fillna(lot_median, inplace = True)

    # Just to be sure we caught all nulls, drop them here
    df = df.dropna()
    return df

# 
def split_data(df):
    ''' 
    This function will take your clean dataframe and split it
    '''
    
    # split the data
    train_validate, test = train_test_split(df, test_size=.2, random_state=177)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=177)
    
    # show the split
    print(f'Dataframe has been split: ')
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    # return train validate and test
    return train, validate, test 