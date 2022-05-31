#%%
import findgi_functions as findgi
import math
from itertools import permutations
import dill
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

iNatCSV,dObs = findgi.getLatestObsCSV()
weatherCSV,dWeather = findgi.getLatestWeatherCSV()

dfF,dfW = findgi.iNatWeatherCSV2df(iNatCSV,weatherCSV)
fungiFamPivot,famcts = findgi.pivotObs(dfF)
fungiAgg,trainWeatherAgg = findgi.fungiWeatherAgg(fungiFamPivot,dfW)
pFungiFam = fungiAgg.divide(fungiAgg.sum(axis=1),axis=0)


# %%
not_logscale = ['precip']
log_scaled = [feat for feat in findgi.features if feat not in not_logscale]

paramGrid = {
    "regressor__learning_rate": [0.05, 0.1],
    "regressor__n_estimators": [30, 40, 50, 80, 100, 120],
    "regressor__max_features": [1,2,3,4],
    "regressor__alpha": [0.1, 0.3, 0.5, 0.7, 0.9]
}

weekRange = 8
nPerms = (int(math.factorial(weekRange)/
        math.factorial(weekRange-
        len(findgi.rollFeatures))))

for i,fam in enumerate(famcts.index[:10]):
    fitScore = 0
    count = 0
    for rollPerm in permutations(range(1,weekRange),4):
        count+=1
        print(f'fam:{i} | {count}/{nPerms}')

        gsGBR = GridSearchCV(
            estimator=findgi.genFamModel(log_scaled,not_logscale),
            param_grid=paramGrid,
            n_jobs=-1,
            cv=TimeSeriesSplit(),
            verbose=1,
            )
        
        X = findgi.rollWeather(trainWeatherAgg,**{'rollFeatures': findgi.rollFeatures,
                            'rollSpans': list(rollPerm)})

        y = pFungiFam[fam]

        gsGBR.fit(X,y)
        print(gsGBR.best_params_)
        fit = gsGBR.score(X,y)
        print(f'fam:{i} | {fit}') 
        if fit>fitScore:
            fitScore = fit
            roll_days = list(rollPerm)

        famparams = {'fam': fam,
                        'famidx': i,
                        'fitScore': fitScore,
                        'bestParams': gsGBR.best_params_,
                        'roll_features': findgi.rollFeatures,
                        'roll_day_span': roll_days}
    dill.dump(famparams, open(f'./data/optRoll/{i:03}_famParams_{fam}.pkd', 'wb'))
    

