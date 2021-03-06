import requests
from bs4 import BeautifulSoup
from geopy.geocoders import Nominatim
import numpy as np
import pandas as pd
from retry import retry
from datetime import datetime
import ast
import os
import glob
from dotenv import load_dotenv
import dill
import shutil
import time
import re

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import GradientBoostingRegressor


#%% globals

fungiStartDate = '2018-01-03'

rollFeatures =  ['temp',
                'humidity',
                'precip',
                'windspeed']

features = rollFeatures + ['week','month']

aggSpan = 'W'

#%% Distance fcns
mile2meter = lambda miles: 1609.344*miles
mile2km = lambda miles: int(mile2meter(miles)/1000)
deg2rad = lambda deg: (deg/180)*np.pi
vdeg2rad = np.vectorize(deg2rad)

def miBWlatlong(latlong,latlongV):
    x = np.cos((latlong[0] + latlongV[:,0])/2)*(latlongV[:,0]-latlong[0])
    y = latlongV[:,1] - latlong[1]
    return np.sqrt(x**2 + y**2)*6371000*0.00062137

#%% API CALLS

#KEYS
load_dotenv()
vcAPIkey = os.environ.get("VISUALCROSSING_API")
SERPapiKey = os.environ.get("SERP_API")
gmAPIkey = os.environ.get("GM_API")

def get_inat_taxonID(query):
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
                'action':'query',
                'format':'json',
                'list':'search',
                'utf8':1,
                'srsearch':query
            }
    data = requests.get(url, params=params).json()
    wikiurl = ('https://en.wikipedia.org/wiki/' +
                data['query']['search'][0]['title'])
    #or see: https://stackoverflow.com/questions/64001004/how-to-get-the-scientific-classification-information-from-wikipedia-page-using
    page = requests.get(wikiurl).text
    soup = BeautifulSoup(page,'lxml')
    parent = soup.find('a',
        attrs={'href':'/wiki/INaturalist','title': 'INaturalist'})
    inat_taxon_id = [a.text for i,a in enumerate(parent.next_elements) if i==2][0]
    return inat_taxon_id

def latlong_from_address(address="Pittsburgh, PA"):
    geolocator = Nominatim(user_agent="JustGetAnAddressApp")
    location = geolocator.geocode(address)
    return np.round(location.latitude,2),np.round(location.longitude,2)

@retry(tries=3, delay=2)
def get_obs_around_address(fungus,
                            address,
                            radius_in_miles,
                            dateStart=None,
                            dateEnd=None):

    lat,lng = latlong_from_address(address)

    taxonField = 'taxon_id'
    if isinstance(fungus,(int,np.integer)):
        inat_taxon_id = fungus
    elif isinstance(fungus,str):
        inat_taxon_id = get_inat_taxonID(fungus)
    elif isinstance(fungus,(list,np.ndarray)):
        taxonField = 'taxon_ids'
        inat_taxon_id = ','.join(map(str,fungus))

    max_results = 200
    baseurl = 'https://api.inaturalist.org/v1/observations'
    params = {'lat': lat,
                'lng': lng,
                'radius': mile2km(radius_in_miles),
                taxonField: inat_taxon_id,
                'per_page': max_results}

    if dateStart:
        params['d1'] = datetime.strptime(
            dateStart,'%Y-%m-%d').astimezone().isoformat()
        if not dateEnd:
            params['d2'] = datetime.now().astimezone().isoformat()
        else:
            params['d2'] = datetime.strptime(
                dateEnd,'%Y-%m-%d').astimezone().isoformat()
    
    obs = []
    first_result = requests.get(baseurl,params=params).json()
    pages = int(np.ceil(first_result['total_results']/max_results))
    obs.append(pd.DataFrame.from_dict(first_result['results']))
    print(f'taxon: {inat_taxon_id}, pages:{pages}')
    for page in range(1,pages):
        params['page'] = page+1
        # print(page+1)
        obs.append(pd.DataFrame.from_dict(requests.get(baseurl,params=params).json()['results']))
    df = pd.concat(obs)
    df['searched_taxon_id'] = inat_taxon_id

    return df

def getWeather(forecast=False,startDate=None,endDate='current',
                    location='Pittsburgh, PA',
                    elements=['datetime','temp', 'humidity', 'precip', 'windspeed'],
                    apiKey=vcAPIkey):
    
    #no dates is 15 day forecast
    if location != 'Pittsburgh, PA':
        location = ','.join(map(str,latlong_from_address(location)))

    if endDate=='current':
        endDate = datetime.now().strftime('%Y-%m-%d')
       
    baseurl = ('https://weather.visualcrossing.com/VisualCrossingWebServices'
              '/rest/services/timeline/')

    params = {
    'unitGroup': 'us',
    'maxDistance': '199558',
    'elements': ','.join(elements),
    'include': 'days',
    'contentType': 'json',
    'key': apiKey
    }

    if forecast:
        resp = requests.get(baseurl + location,params=params).json()
    else:
        date = (startDate + '/' + endDate if startDate else endDate)
        resp = requests.get(baseurl + location + '/' + date,params=params).json()

    df = pd.DataFrame(resp['days'])
    df['latitude'] = resp['latitude']
    df['longitude'] = resp['longitude']
    df['resolvedAddress'] = resp['resolvedAddress']
    df['address'] = resp['address']
    df['timezone'] = resp['timezone']
    df['tzoffset'] = resp['tzoffset']

    return df


def getFungiImgURL(familyName,location='Pennsylvania',apiKey=SERPapiKey):
    
    if os.path.exists(os.path.join(os.getcwd(),
                        'data','images',familyName + '.pkl')):
        return dill.load(open(os.path.join(os.getcwd(),
                        'data','images',familyName + '.pkl'),'rb'))
    
    baseurl = 'https://serpapi.com/search.json'

    params = {
    "q": familyName + location,
    "tbm": "isch",
    "ijn": "0",
    "api_key": SERPapiKey
    }

    resp = requests.get(baseurl,params=params).json()
    dill.dump(resp,open(os.path.join(os.getcwd(),
                        'data','images',familyName + '.pkl'),'wb'))

    return resp


def getFungiFamImg(imgSearchResponse,fam):
    #response from getFungiImgURL
    if os.path.exists(os.path.join(os.getcwd(),
                'data','images',f'{fam}_0.jpeg')):
        print(f'got image files for {fam}')
    else:
        for i,el in enumerate(imgSearchResponse['images_results'][:6]):
            imgURL = el['thumbnail']

            # Open the url image, set stream to True, this will return the stream content.
            r = requests.get(imgURL, stream = True)

            # Check if the image was retrieved successfully
            if r.status_code == 200:
                # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
                r.raw.decode_content = True
                
                
                # Open a local file with wb ( write binary ) permission.
                with open(os.path.join(os.getcwd(),
                    'data','images',f'{fam}_{i}.jpeg'),'wb') as f:
                    
                    shutil.copyfileobj(r.raw, f)
                    
                print('Image sucessfully Downloaded: ',f'{fam}_{i}.jpeg')
            else:
                print('Image Couldn\'t be retreived')


@retry(tries=3, delay=2)
def getObsImg(obs):
    baseurl = 'https://api.inaturalist.org/v1/observations'
    resp = requests.get(baseurl + '/' + str(obs)).json()
    return {'obs': [i['url'] for i in resp['results'][0]['photos']],
    'taxon': resp['results'][0]['taxon']['default_photo']['medium_url']}


@retry(tries=3, delay=2)
def getTaxonIDinfo(taxonID,delayAfter=0):
    '''
    Gets iNaturalist taxonID data associated with provided
    low level taxonID. Data includes image URLs.
    '''

    fName = os.path.join(os.getcwd(),
            'data','taxa',f'taxonID_{taxonID}.pkl')

    if os.path.exists(fName):
            resp = dill.load(open(fName,'rb'))
    else:
        baseurl = 'https://api.inaturalist.org/v1/taxa/'
        resp = requests.get(baseurl + str(taxonID)).json()
        dill.dump(resp,open(fName,'wb'))

        if delayAfter>0:
            time.sleep(delayAfter)

    return resp


@retry(tries=3, delay=2)
def getFamTaxonID(fungiFam,delayAfter=0):
    '''
    Gets iNaturalist taxonID data associated with provided
    fungus family. Data includes image URLs

      "name": "Sphinctrinaceae",
      "rank": "family",
      "extinct": false,
      "id": 174981, <-- FAM TAXON ID

    '''
    fName = os.path.join(os.getcwd(),
            'data','taxa',f'famTaxon_{fungiFam}.pkl')
    
    if os.path.exists(fName):
            resp = dill.load(open(fName,'rb'))
    else:
        baseurl = 'https://api.inaturalist.org/v1/taxa'
        params = {
            "q": fungiFam,
            "rank": "family"
            }

        resp = requests.get(baseurl,params=params).json()
        dill.dump(resp,open(fName,'wb'))

        if delayAfter>0:
            time.sleep(delayAfter)
    
    return resp

#%% DATA PREP

def cleanScrapedObs(iNatScrapeCSV):
    df_scrape = pd.read_csv(iNatScrapeCSV,low_memory=False)
    df_scrape['id'] = df_scrape['id'].astype(int)
    df_scrape['idQuality'] = df_scrape['quality_grade']
    df_scrape = df_scrape.drop(columns=df_scrape.columns[
        ~np.isin(df_scrape.columns,
        ['id','taxon','geojson','observed_on',
        'created_at','searched_taxon_id','idQuality','user'])])

    df_scrape['inaturalistLogin'] = df_scrape['user'].apply(lambda x: ast.literal_eval(x)['login'])
    df_scrape['taxonID'] = df_scrape['taxon'].apply(lambda x: ast.literal_eval(x)['id'])
    df_scrape['scientificName'] = df_scrape['taxon'].apply(lambda x: ast.literal_eval(x)['name'])
    df_scrape['rank'] = df_scrape['taxon'].apply(lambda x: ast.literal_eval(x)['rank'])
    df_scrape['decimalLongitude'] = df_scrape['geojson'].apply(lambda x: ast.literal_eval(x)['coordinates'][0])
    df_scrape['decimalLatitude'] = df_scrape['geojson'].apply(lambda x: ast.literal_eval(x)['coordinates'][1])
    df_scrape = df_scrape.drop(columns=['taxon','geojson'])

    #use created_at for those missing 'observed_on'
    df_scrape['observed_on'][df_scrape['observed_on'].isna()] = \
        df_scrape['created_at'][df_scrape['observed_on'].isna()]
    df_scrape = df_scrape.drop(columns=['created_at'])

    dists = df_scrape[['decimalLatitude','decimalLongitude']].to_numpy()
    df_scrape['dist_from_PGH'] = miBWlatlong(vdeg2rad(latlong_from_address('Pittsburgh, PA')),
        vdeg2rad(dists))
    df_scrape['eventDate'] = pd.to_datetime(pd.to_datetime(df_scrape.observed_on,utc=True).dt.date)
    df_scrape = df_scrape.drop(columns=['observed_on'])
    df_scrape['source'] = 'web_scrape'

    df_scrape[['family','genus']] = None
    df_scrape['species'] = df_scrape.scientificName.str.split(" ",1,expand=True).iloc[:,1]

    keepcol = ['id','inaturalistLogin','eventDate','dist_from_PGH',
    'decimalLatitude','decimalLongitude',
    'taxonID','searched_taxon_id',
    'family','genus','species','scientificName','rank',
    'source','idQuality']

    df_scrape = df_scrape[keepcol]
    #still need to get rid of dupe obs IDs and fill in KPCOFGS

    return df_scrape


def getLatestFileByFMod(fGlobPath):
    F = glob.glob(fGlobPath)
    latestModF = (max(F, key=os.path.getctime) if F else None)

    return latestModF


def getLatestFileByFileNameDate(fGlobPath):
    F = glob.glob(fGlobPath)
    if not F:
        return None
    dates = [datetime.strptime(
        re.search(
            r'(?P<date>\d{4}-\d{2}-\d{2})\.[a-z]+?',el)['date'],'%Y-%m-%d')
             for el in F]
    
    latestDateFidx = np.argmax(dates)
    return F[latestDateFidx],dates[latestDateFidx]


def getLatestScrapeCSV(): 
    return getLatestFileByFileNameDate(
        os.path.join(os.getcwd(),'data','obs_fungi_web_PGH*.zip')
    )


def getLatestTaxonKey():
    return getLatestFileByFileNameDate(
        os.path.join(os.getcwd(),'data','taxonKey*.pkl')
    )


def getLatestObsCSV():
    return getLatestFileByFileNameDate(
        os.path.join(os.getcwd(),'data','*fungi_web-w-db*.zip')
    )


def getLatestWeatherCSV():
    return getLatestFileByFileNameDate(
        os.path.join(os.getcwd(),'data','*weather*.zip')
    )

def getLatestModelFile():
    return getLatestFileByFileNameDate(
        os.path.join(os.getcwd(),'data','*models*.pk')
    )


def weatherCSV2df(weather_csv):
    dfW = pd.read_csv(weather_csv,low_memory=False)
    dfW['datetime'] = pd.to_datetime(dfW.datetime)
    dfW = dfW.set_index('datetime')

    return dfW


def iNatCSV2df(iNat_csv,startDate=fungiStartDate):
    dfF = pd.read_csv(iNat_csv,low_memory=False)
    dfF.eventDate = pd.to_datetime(dfF.eventDate)
    dfF = dfF[dfF.eventDate.gt(startDate)]
    dfF = dfF.set_index('eventDate',drop=False)
    dfF.index.name = 'date'

    return dfF

def iNatWeatherCSV2df(iNat_csv,weather_csv,startDate=fungiStartDate):
    '''
    Take cleaned iNat CSV and raw weather CSV,
    output df for each.
    '''
    dfF = iNatCSV2df(iNat_csv,startDate)
    dfW = weatherCSV2df(weather_csv)

    return dfF,dfW

def pivotObs(dfFungi,fillMissingDates=True):
    """
    Take weather and iNat obs dfs and output a pivot table of
    fungi family obs counts with weather columns.
    """
    dfF_fam = (pd.DataFrame(
        dfFungi[['eventDate','family']]
        .value_counts())
        .reset_index()
        .pivot(index='eventDate',columns='family',values=0)
        .fillna(0))
    famcts = dfF_fam.sum().sort_values(ascending=False)
    
    dfF_fam = dfF_fam.join(
        dfFungi['inaturalistLogin'].groupby(
            level='date').nunique().rename('n_unique_users'))
    
    if fillMissingDates:
        #check for missing dates:
        full_date_range = pd.date_range(
            start=dfF_fam.index.min(),
            end=dfF_fam.index.max(),freq="D"
        )
        #fill missing dates with count 0:
        dfF_fam = dfF_fam.reindex(full_date_range,fill_value=0)

    return dfF_fam,famcts


def fungiWeatherAgg(fungiFamPivot,weather,
                    aggSpan=aggSpan,
                    weatherFeatures=rollFeatures):
    # params = {}
    # for parameter, value in kwargs.items():
    #     params[parameter] = value

    fams = [c for c in fungiFamPivot.columns if 'n_unique_users' not in c]
    fungiFamWeatherPivot = fungiFamPivot.join(weather[weatherFeatures])

    if aggSpan=='W':
        fungiAgg = fungiFamWeatherPivot[fams].groupby(
                            pd.Grouper(freq='W')).agg(sum)

        weatherAgg = fungiFamWeatherPivot[weatherFeatures].groupby(
                            pd.Grouper(freq='W')).agg(np.mean)
    else:
        fungiAgg = fungiFamWeatherPivot[fams].groupby(
                    pd.Grouper(freq=str(aggSpan)+ 'D')).agg(sum)

        weatherAgg = fungiFamWeatherPivot[weatherFeatures].groupby(
                            pd.Grouper(freq=str(aggSpan)+ 'D')).agg(np.mean)
  
    weatherAgg['month'] = weatherAgg.index.month
    weatherAgg['week'] = weatherAgg.index.isocalendar().week

    return fungiAgg,weatherAgg


def rollWeather(weatherAgg,**kwargs):
    params = {}
    for parameter, value in kwargs.items():
        params[parameter] = value

    not_rolled = [f for f in weatherAgg.columns if f not in params['rollFeatures']]
    
    for i,feature in enumerate(params['rollFeatures']):
        weatherAgg['roll_' + feature] = (
            weatherAgg[feature]
            .rolling(params['rollSpans'][i],
            min_periods=1)
            .mean())
    
    weatherRoll = weatherAgg[[c for c in weatherAgg.columns if 'roll' in c]]
    weatherRoll = weatherRoll.rename(columns={k:v for k,v in zip(weatherRoll.columns,
                    [s.strip('roll_') for s in weatherRoll.columns])})
  
    if not_rolled:
        weatherRoll[not_rolled] = weatherAgg[not_rolled]

    return weatherRoll


def weatherAggAndRoll(df_weather,
                      rollFeatures=rollFeatures,
                      rollSpans=[0,0,0,0],
                      aggSpan='W',
                      startDate=fungiStartDate,
                      ):

    df_weather = df_weather.iloc[df_weather.index>fungiStartDate,:]
    
    if aggSpan=='W':
        weatherAgg = df_weather.groupby(pd.Grouper(freq='W')).agg(np.mean)
    else:
        weatherAgg = df_weather.groupby(
                            pd.Grouper(freq=str(aggSpan)+ 'D')).agg(np.mean)

    weatherAgg['month'] = weatherAgg.index.month
    weatherAgg['week'] = weatherAgg.index.isocalendar().week
    
    return rollWeather(weatherAgg,**{'rollFeatures': rollFeatures,
                            'rollSpans': rollSpans})


def getRollParams(paramDirectory):
    
    famParams = []
    for f in glob.glob(os.path.join(paramDirectory,'*_*.pkd')):
        famParams.append(dill.load(open(f, 'rb')))

    dfParams = pd.DataFrame(famParams)
    dfParams[famParams[0]['roll_features']] = dfParams.roll_day_span.to_list()

    return dfParams


def addWeekMonthCols(df):    
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    return df


def updateWeather(oldWeatherCSV=None,saveFile=True,
                  elements=['datetime','temp', 'humidity', 'precip', 'windspeed']):
    if not oldWeatherCSV:
        fs = glob.glob(os.path.join(os.getcwd(),'data','*weather*.zip'))
        oldWeatherCSV = max(fs, key=os.path.getctime)
    
    dfWold = pd.read_csv(oldWeatherCSV,low_memory=False)
    dfWold['datetime'] = pd.to_datetime(dfWold.datetime)
    dfWold = dfWold.set_index('datetime')

    if dfWold.index[-1].strftime("%Y-%m-%d") == datetime.now().strftime("%Y-%m-%d"):
        return dfWold

    startDate = dfWold.index[-1].strftime('%Y-%m-%d')   

    dfWnew = getWeather(startDate=startDate,elements=elements)
    dfWnew['datetime'] = pd.to_datetime(dfWnew.datetime)
    dfWnew = dfWnew.set_index('datetime')

    dfW = dfWold.append(dfWnew)[[i for i in elements if i != 'datetime']]
    dfW = dfW[~dfW.index.duplicated(keep='last')]
    
    if saveFile:
        dfW['datetime'] = dfW.index
        fName = (f'{dfWnew.address[-1]}_weather_{dfWold.index[0].strftime("%Y-%m-%d")}'
                  f'_to_{dfW.index[-1].strftime("%Y-%m-%d")}')

        compression_opts = dict(method='zip',
                        archive_name=fName + '.csv')  
        dfW.to_csv(os.path.join(os.getcwd(),'data',
                fName + '.zip'), index=False,
                compression=compression_opts)
        
        os.remove(oldWeatherCSV)

    return dfW.drop(columns='datetime')


def catForecast(dfWeather):
    forecast = getWeather(forecast=True)
    dfFuture = pd.concat([dfWeather,forecast.set_index('datetime')])[dfWeather.columns]
    dfFuture.index = pd.to_datetime(dfFuture.index)

    return dfFuture,forecast


def iNatObsScrape(radius=100,reqWaitTime=0.7):
    '''
    Looks for most recent web scrape csv and scrapes iNat by fungi family
    from that date on.
    '''
    taxonKey = dill.load(open(getLatestTaxonKey()[0],'rb'))
    latestScrapeF,dScraped = getLatestScrapeCSV()

    uTaxons = taxonKey.family_taxonID.unique()
    dateStart = dScraped

    newstart = 0
    total = len(uTaxons)
    total = total-newstart
    taxretry = []
    dfs = []
    for i,taxon in enumerate(uTaxons[newstart:]):
        print(f'{i}/{total}')
        try:
            dfs.append(get_obs_around_address(taxon,
                                'Pittsburgh, PA',
                                radius,
                                dateStart=dateStart))
        except:
            print(f'could not get {taxon}')
            taxretry.append(taxon)
        time.sleep(reqWaitTime)

    df_scrape = pd.concat(dfs)

    dateEnd = datetime.now().strftime('%Y-%m-%d')
    fName = f'obs_fungi_web_PGH_{radius}mi_radius_{dateStart}_{dateEnd}'

    compression_opts = dict(method='zip',
                            archive_name=fName + '.csv')  
    df_scrape.to_csv(os.path.join(os.getcwd(),'data',
            fName + '.zip'), index=False,
            compression=compression_opts)
    
    return df_scrape,taxretry


def updateObsWithScrape():   

    fScrape,dScrape = getLatestScrapeCSV()
    fObs,dObs = getLatestObsCSV()

    if dScrape <= dObs:
        pass 
    else:
        dfs_clean = cleanScrapedObs(fScrape)
        dfs_clean.eventDate = pd.to_datetime(dfs_clean.eventDate)
        dfs_clean = dfs_clean.set_index('eventDate',drop=False)
        dfs_clean = dfs_clean.sort_index()
        #only keep genus, species rank IDs
        dfs_clean = dfs_clean[dfs_clean['rank'].isin(['genus','species'])]

        dff = iNatCSV2df(getLatestObsCSV()[0])
        dff = dff.sort_index()
        dff = dff[~np.isin(dff.id,dfs_clean.id)]

        taxonKey = dill.load(open(getLatestTaxonKey()[0],'rb'))

        missingTaxa = dfs_clean.taxonID[~np.isin(dfs_clean.taxonID,taxonKey.taxonID.unique())].unique()
        
        if len(missingTaxa>0):
            delay = (0 if len(missingTaxa)<60 else 0.7)
            dMissingTaxa = {tx:getTaxonIDinfo(tx,delay) for tx in missingTaxa}
            getKeys = ['name', 'rank', 'preferred_common_name' , 'default_photo', 'ancestor_ids']
            dLowTaxa = {}
            for k,v in dMissingTaxa.items():
                dLowTaxa[k] = {kk:v['results'][0].get(kk,'') for kk in getKeys}
            
            lowTaxaDF = pd.DataFrame(dLowTaxa).T
            lowTaxaDF['img'] = lowTaxaDF.default_photo.apply(lambda x: x['medium_url'])
            lowTaxaDF.drop(columns=['default_photo'],inplace=True)
            lowTaxaDF = lowTaxaDF.reset_index().rename(columns={'index':'taxonID'})
            lowTaxaDF['species'] = lowTaxaDF.apply(lambda x: (x['name'].split()[1] if x['rank']=='species' else 'NA'),axis=1)
            lowTaxaDF['genus'] = lowTaxaDF.apply(lambda x: (x['name'].split()[0] if x['rank']=='species' else x['name']),axis=1)
        
            fams = taxonKey[['family', 'family_taxonID', 'famImg']].drop_duplicates()

            famTaxa = lowTaxaDF.ancestor_ids.apply(lambda x:
                x[np.max(np.where(np.isin(x,fams.family_taxonID)))])
            
            fams[fams.family_taxonID.isin(famTaxa)]

            lowTaxaDF = lowTaxaDF.join(pd.concat(
                [fams[fams.family_taxonID.eq(i)] for i in famTaxa]).reset_index(drop=True))

            taxonKeyUpdate = pd.concat([taxonKey,lowTaxaDF])
            taxonKeyUpdate.drop_duplicates(inplace=True)
            taxonKeyUpdate.drop_duplicates(subset='taxonID',inplace=True)
            dill.dump(taxonKeyUpdate,open(os.path.join(os.getcwd(),
                'data',f'taxonKey_{datetime.now().strftime("%Y-%m-%d")}.pkl'),'wb'))
            taxonKey = taxonKeyUpdate

            
        famgenus = pd.concat([taxonKey[taxonKey.taxonID.eq(i)][['family','genus']] 
                for i in dfs_clean.taxonID.values]).reset_index(drop=True)
        dfs_clean['family'] = famgenus.family.values
        dfs_clean['genus'] = famgenus.genus.values
        dfs_clean = dfs_clean.drop('rank',axis=1)

        dffUpdate = pd.concat([dff[dfs_clean.columns],dfs_clean])

        fName = re.sub(r'(.*)_web_(.*)\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2}).zip',
                    r'\1_web-w-db_\2\3',os.path.split(fScrape)[-1])

        compression_opts = dict(method='zip',
                            archive_name=fName + '.csv')  

        dffUpdate.to_csv(os.path.join(os.getcwd(),'data',
                fName + '.zip'), index=False,
                compression=compression_opts)

        return dffUpdate



def fungiFamFromQuery(query,taxonKey=None):
    if not isinstance(taxonKey,pd.DataFrame):
        taxonKey = dill.load(open(getLatestTaxonKey()[0],'rb'))
    
    try:
        fam = taxonKey[taxonKey.preferred_common_name.str.contains(
            query,case=False)].family.mode()[0]
    except:
        try:
            taxonID = int(get_inat_taxonID((query + ' mushroom' if 'mushroom' not in query else query)))
            fam = taxonKey[taxonKey.taxonID.eq(taxonID)].family.item()
        except:
            return None
    return fam


#%% TRAINING AND PREDICTIONS
def genFamModel(feat_logScale,feat_notLogScale,fit_params=None):
    log_scale_transformer = FunctionTransformer(np.log, validate=False)

    featureProcessor = ColumnTransformer(
        transformers=[
            ("log", log_scale_transformer, feat_logScale),
            ("regular", 'passthrough', feat_notLogScale),
        ]
    )

    if fit_params:
        fit_params['random_state'] = 42
        GBR_pipe = Pipeline(
            [
                ('preprocessor',featureProcessor),
                ('scale',StandardScaler()),
                ('regressor', GradientBoostingRegressor(**fit_params))
            ]
        )
    else:
        GBR_pipe = Pipeline(
            [
                ('preprocessor',featureProcessor),
                ('scale',StandardScaler()),
                ('regressor', GradientBoostingRegressor(random_state=42))
            ]
        )

    return GBR_pipe


def getFamModels(dfRollParams,pFungiFam,weatherAgg,
                 features = features,
                 rollFeatures = rollFeatures,
                 not_logscale = ['precip']):
    
    log_scaled = [feat for feat in features if feat not in not_logscale]

    famModels = {}
    testData = {}
    #generate models for each family
    for i,r in dfRollParams.iterrows():
        famModels[r.fam] = genFamModel(log_scaled,not_logscale)

        X = rollWeather(weatherAgg[features],
                            **{'rollFeatures': rollFeatures,
                            'rollSpans': r.roll_day_span})

        y = pFungiFam[r.fam]

        tscv = TimeSeriesSplit(n_splits=2)
        trainIDX, testIDX = list(tscv.split(X, y))[-1]
        
        famModels[r.fam].fit(X[trainIDX],y[trainIDX])
        testData[r.fam] = (X[testIDX],y[testIDX])

    return famModels,testData


def getTrainingData(iNat_csv,weather_csv,rollParamDir):
    dfF,dfWmodels = iNatWeatherCSV2df(iNat_csv,weather_csv)
    fungiFamPivot,_ = pivotObs(dfF)
    fungiAgg,trainWeatherAgg = fungiWeatherAgg(fungiFamPivot,dfWmodels)
    pFungiFam = fungiAgg.divide(fungiAgg.sum(axis=1),axis=0)

    dfRollParams = getRollParams(rollParamDir)

    return dfRollParams,pFungiFam,trainWeatherAgg


def trainObsWeather2models(iNat_csv,weather_csv,rollParamDir):
    
    dfRollParams,pFungiFam,trainWeatherAgg = getTrainingData(
                                                iNat_csv,weather_csv,rollParamDir)

    famModels,testData = getFamModels(dfRollParams,pFungiFam,trainWeatherAgg)

    return famModels,dfRollParams,pFungiFam,trainWeatherAgg,testData


def getLikelyFams(famModels,avWeatherWeek):
    fProbs = {}
    for k,v in famModels.items():
        fProbs[k] = v.predict(avWeatherWeek)[0]
    df = pd.DataFrame.from_dict([fProbs]).T.reset_index().rename(
        columns={'index':'family',0: 'p(Fam | Env)'})
    df = df.sort_values(by=['p(Fam | Env)'],ascending=False)

    return df
