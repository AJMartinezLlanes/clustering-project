# Clustering Project
by AJ Martinez

## Plan
 - State project goals
 - State initial questions
 - Acquire data
 - Prepare Data
 - Create Functions
 - Set the data context
 - Answer questions 
 - Establish a baseline
 - Fit models
 - Recomendations
 - Next steps

## Data Dictionary
|Variable|Description|
|---|---|
| parcelid  | Parcel Identification Number  |       
| bathroomcnt | Number of Bathrooms|
| bedroomcnt  | Number of Bedrooms|
| calculatedfinishedsquarefeet | Property Square Footage|
| fips | Federal Information Processing Standard  |
| latitude | Latitude|               
| longitude | Longitude|      
| lotsizesquarefeet | Lot Square Footage |
| regionidcounty | County Region Identification|           
| regionidzip | Zip Code |
| yearbuilt | Year Property was built|
| structuretaxvaluedollarcnt | Structure Tax Value|
| taxvaluedollarcnt | Property Tax Value |
| landtaxvaluedollarcnt | Land Tax Value|
| taxamount | Tax Paid|
| logerror | Zestimate error|
| transactiondate | Transaction date|
| propertylandusedesc | Land Use Description |
| county | County |


## Steps to reproduce
- Notebooks with code are in this repo 
- wrangle.py with functions on how to acquire data clean and prep data
- Notebook with a walkthrough of the findings
- Description of the steps taken


## Project Goals
- Explore Single Family Properties using attributes of the properties.
- Use Clusters to find relationships
- Find the key drivers of log error on single family properties. 
- Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
- Make recommendations on what works or doesn't work with data and models



## Initial Questions
- Is there a relationship between Log Error and Zip Code?
- Is there a relationship between Log Error and Square Footage?
- Is there a relationship between Log Error and Tax Value?
- Is there a relationship between Log Error and Latitude?

## Deliberables
1. Readme(.md)
2. Final Report(.ipynb)
3. Wrangle(.py) with functions
4. 1+ non-final notebooks (.ipynb) created while working the project, containing exploration and modeling work
