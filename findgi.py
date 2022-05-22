#%%
import findgi_functions as findgi
import numpy as np
import pandas as pd
from bokeh.io import curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import (Div, ColumnDataSource,TextInput, Select,LinearAxis, 
                          DataRange1d, Select, IndexFilter, CDSView, GMapOptions)
from bokeh.plotting import figure, gmap
import colorcet as cc
import seaborn as sns
from matplotlib.colors import to_hex
import os
import dill
import re

#%% load data files
taxonKey = dill.load(open(findgi.getLatestTaxonKey()[0],'rb'))

weatherCSV,dWeather = findgi.getLatestWeatherCSV()
iNatCSV,dObs = findgi.getLatestObsCSV()
modelPkl,dModel = findgi.getLatestModelFile()

modelParamsDir = os.path.join(os.getcwd(),'data','modelParams')

#%%
# load models
if not modelPkl or dObs>dModel:
    famModels,dfRollParams,pFungiFam,trainWeatherAgg = \
        findgi.trainObsWeather2models(
        iNatCSV,weatherCSV,modelParamsDir)

    d = {v:eval(v) for v in ['iNatCSV','weatherCSV',
                            'famModels','dfRollParams',
                            'pFungiFam','trainWeatherAgg']}
    dill.dump(d,open(os.path.join(os.getcwd(),'data',
        "models_{0}.pk".format(re.search(r'\d{4}-\d{2}-\d{2}',iNatCSV)[0])),'wb'))

else:
    locals().update(dill.load(open(modelPkl,'rb')))

dfW = findgi.addWeekMonthCols(findgi.updateWeather())
weekWeather = pd.DataFrame(dfW.iloc[-7:,:].agg(np.mean)).T

dfMix = findgi.getLikelyFams(famModels,weekWeather)
famChoices = dfMix.family.to_list()

#%% INIT FAM DATA
initMushroom = 'morel'

fam = findgi.fungiFamFromQuery(initMushroom)

rollParams = dfRollParams[dfRollParams.fam.eq(fam)].roll_day_span.item()
existing = pFungiFam[fam]

#modeled
rollWeatherAgg = findgi.weatherAggAndRoll(dfW,
                    rollFeatures=findgi.rollFeatures,
                    rollSpans=rollParams,
                    aggSpan='W')
modeled = pd.Series(data=famModels[fam].predict(rollWeatherAgg),
                    index=rollWeatherAgg.index)

dfWfuture,_ = findgi.catForecast(dfW)

rollWeatherAggF = findgi.weatherAggAndRoll(dfWfuture,
                    rollFeatures=findgi.rollFeatures,
                    rollSpans=rollParams,
                    aggSpan='W')    
future = pd.Series(data=famModels[fam].predict(rollWeatherAggF),
                    index=rollWeatherAggF.index)

dfFam = pd.concat([existing,modeled,future],axis=1).rename(columns=
        {fam: 'existing',0:'modeled',1:'future'})

addDates = {'existing': dfFam.shape[0]-pFungiFam.shape[0],
            'modeled':dfFam.shape[0]-rollWeatherAgg.shape[0]}

#%% given fam plot
oneFamSource = ColumnDataSource(dfFam)

oneFamFig = figure(plot_width=600, plot_height=300,
    title='Probability of finding fungus family given environment:',
    x_axis_type = 'datetime',
    x_axis_label = 'Date',
    y_axis_label = 'P( Family | Environment )',
    tools="pan,box_zoom,wheel_zoom,reset",
    )

oneFamFig.line('index', 'existing', legend_label='data', source=oneFamSource, 
        line_color='black',name='ticker1',line_width=3)
oneFamFig.line('index', 'future', legend_label='prediction', source=oneFamSource,
        line_dash='dotted',line_color='green',name='ticker1',line_width=3)
oneFamFig.legend.location = 'top_left'
oneFamFig.extra_y_ranges = {"rel": DataRange1d(end=1)}
oneFamFig.add_layout(LinearAxis(y_range_name="rel"), 'right')
oneFamFig.title.text_font_size = '14pt'


#%% image plot

dfFamImg = taxonKey[['family','famImg']].drop_duplicates()
famImg = figure(plot_width=200, plot_height=150, tools="")
source_image = ColumnDataSource(data={})
source_image.data = {'famImg': [dfFamImg[dfFamImg.family.eq(fam)].famImg.item()]}

famImg.image_url("famImg",source=source_image,x=0, y=1, w=0.8, h=0.6)
famImg.xaxis.visible = False
famImg.yaxis.visible = False
famImg.xgrid.visible = False
famImg.ygrid.visible = False


#%% variety plot
#TODO: current, next week, week after that
dfMix = dfMix.iloc[:15,:]
dfMix['palette'] = sns.color_palette(cc.glasbey, n_colors=len(dfMix))
dfMix['palette'] = dfMix['palette'].apply(to_hex)
dfMix = dfMix.merge(dfFamImg,on='family')
dfMix['pFam'] = dfMix['p(Fam | Env)'].round(5).astype(str)

varietySource = ColumnDataSource(dfMix.iloc[::-1])

famVarietyTooltips = """
    <div>
        <div>
            <img
                src="@famImg" height="90" alt="@famImg" width="90"
                style="float: left; margin: 0px 0px 0px 0px;"
                border="0"
            ></img>
            <div>
                <span style="font-size: 14px; font-weight: bold;">@pFam</span>
            </div>
        </div>
    </div>
"""

famVariety = figure(y_range=varietySource.data['family'], height=450, title="Top 15 most likely fungi families this week:",
           toolbar_location=None, tooltips=famVarietyTooltips, tools="",x_axis_location="above")
famVariety.hbar(y="family", right="p(Fam | Env)", width=1.5,
       source = varietySource,
       color='palette', 
       )
famVariety.xaxis.axis_label = 'P( Family | Environment )'
famVariety.xgrid.grid_line_color = None
famVariety.title.text_font_size = '14pt'


#%% MAP:
dff = findgi.iNatCSV2df(findgi.getLatestObsCSV()[0])
dff = dff[['taxonID','decimalLatitude','decimalLongitude']]
dff['dateFancy'] = dff.index.strftime('%b %d, %Y')

#get info from taxonKey
dff = dff.merge(taxonKey[['taxonID','family','name','preferred_common_name','img']],
    on='taxonID')

mapSource = ColumnDataSource(data=dff)

view = CDSView(source=mapSource, filters=[IndexFilter(
    list(np.where(dff.family.eq('Morchellaceae'))[0]))])

mapToolTips = """
    <HTML>
    <HEAD>
        <style>
            .column {
                float: left;
                width: 50%;
                }

            .left {
            width: 10%;
            }

            .right {
            width: 90%;
            }
            
            /* Clear floats after the columns */
            .row:after {
            content: "";
            display: table;
            clear: both;
            }

        </style>
    </HEAD>
    <BODY>
    <div class="row">
        <div class="column left">
            <img
                src="@img" height="90" alt="@img" width="90"
                style="float: left; margin: 0px 0px 0px 0px;"
                border="0"
            ></img>
        </div>
        <div class="column right">
            <div class="row">
                <span style="font-size: 12px; font-weight: bold;">Date:</span>
                <span style="font-size: 12px;">@dateFancy</span>
            </div>
            <div class="row">
                <span style="font-size: 12px; font-weight: bold;">lat, lng:</span>
                <span style="font-size: 12px;">@decimalLatitude, @decimalLongitude</span>
            </div>
            <div class="row">
                <span style="font-size: 12px; font-weight: bold;">common name:</span>
                <span style="font-size: 12px;">@preferred_common_name</span>
            </div>
            <div class="row">
                <span style="font-size: 12px; font-weight: bold;">name (Genus species):</span>
                <span style="font-size: 12px;">@name</span>
            </div>
        </div>
    </div>
    </BODY>
    </HTML>
"""


lat,lng = findgi.latlong_from_address()
map_options = GMapOptions(lat=lat, lng=lng, map_type="roadmap", zoom=8)
mapFig = gmap(findgi.gmAPIkey, map_options,
        title="Mapped observations of fungus family:", tooltips=mapToolTips,
        tools="pan,wheel_zoom,reset,hover")
mapFig.circle(x="decimalLongitude", y="decimalLatitude", 
            size=7, fill_color="blue", fill_alpha=0.4,
            source=mapSource, view=view)
mapFig.title.text_font_size = '14pt'


#%% WIDGETS:
text_input = TextInput(value=initMushroom, title="Enter mushroom (eg. honey, oyster, chanterelle):")
select = Select(title="OR choose fungus family (ordered by probability):", value=fam, options=famChoices)

# WIDGET CALLBACKS:
def famFromTextInput(attrname, old, query):
    global select
    if query == "" or 'not found' in query:
        return

    fam = findgi.fungiFamFromQuery(query.lower())
    if fam is None:
        text_input.value = f"{query} not found..."
        return
    #prevent bouncing
    select.remove_on_change('value',getFamProb)
    select.value = fam
    getFamProb('textInput', old, fam)
    select.on_change('value',getFamProb)

def getFamProb(attrname, old, fam):
    global dfWfuture,dfW,dfRollParams
    if attrname != 'textInput':
        text_input.value = ""
    
    rollParams = dfRollParams[dfRollParams.fam.eq(fam)].roll_day_span.item()
    oneFamSource.data['existing'] = np.pad(pFungiFam[fam].values,
                            (0,addDates['existing']),'constant',constant_values=np.nan)

    #modeled
    rollWeatherAgg = findgi.weatherAggAndRoll(dfW,
                        rollFeatures=findgi.rollFeatures,
                        rollSpans=rollParams,
                        aggSpan='W')
    oneFamSource.data['modeled'] = np.pad(famModels[fam].predict(rollWeatherAgg),
                            (0,addDates['modeled']),'constant',constant_values=np.nan)
    rollWeatherAggF = findgi.weatherAggAndRoll(dfWfuture,
                      rollFeatures=findgi.rollFeatures,
                      rollSpans=rollParams,
                      aggSpan='W')    
    oneFamSource.data['future'] = famModels[fam].predict(rollWeatherAggF)

    try:
        source_image.data['famImg'] = [dfFamImg[dfFamImg.family.eq(fam)].famImg.item()]
    except:
        pass
    view.filters[0] = IndexFilter(
        list(np.where(dff.family.eq(fam))[0]))


text_input.on_change('value', famFromTextInput)
select.on_change('value',getFamProb)

desc = Div(text=open("./description.html").read(), width=800, sizing_mode="stretch_width")

inputs = column(text_input,select)

curdoc().add_root(layout([
[
column(
    desc,row(inputs,famImg),mapFig
),
column(
    oneFamFig,famVariety
)
]
]))

curdoc().title = "findgi"