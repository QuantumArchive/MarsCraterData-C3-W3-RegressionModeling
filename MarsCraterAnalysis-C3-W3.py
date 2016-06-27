# coding: utf-8

"""
Created on Tue June 26 14:43:38 2016

@author: Chris
"""
import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats

#from IPython.display import display
#get_ipython().magic(u'matplotlib inline')

#bug fix for display formats to avoid run time errors
pandas.set_option('display.float_format', lambda x:'%f'%x)

#Set Pandas to show all columns in DataFrame
pandas.set_option('display.max_columns', None)
#Set Pandas to show all rows in DataFrame
pandas.set_option('display.max_rows', None)

#data here will act as the data frame containing the Mars crater data
data = pandas.read_csv('D:\\Coursera\\marscrater_pds.csv', low_memory=False)

#convert the latitude and diameter columns to numeric and ejecta morphology is categorical
data['LATITUDE_CIRCLE_IMAGE'] = pandas.to_numeric(data['LATITUDE_CIRCLE_IMAGE'])
data['DIAM_CIRCLE_IMAGE'] = pandas.to_numeric(data['DIAM_CIRCLE_IMAGE'])
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].astype('category')

#Any crater with no designated morphology will be replaced with NaN
data['MORPHOLOGY_EJECTA_1'] = data['MORPHOLOGY_EJECTA_1'].replace(' ',numpy.NaN)

#Here we will subset out craters with the three ejecta morphologies we are interested in
morphofinterest = ['Rd','SLEPS','SLERS']
data = data.loc[data['MORPHOLOGY_EJECTA_1'].isin(morphofinterest)]
data.head(5)

#We now center the data
data['CENTERED_LATITUDE'] = (data['LATITUDE_CIRCLE_IMAGE'] - data['LATITUDE_CIRCLE_IMAGE'].mean())
data['CENTERED_LONGITUDE'] = (data['LONGITUDE_CIRCLE_IMAGE'] - data['LONGITUDE_CIRCLE_IMAGE'].mean())

#We now look at our data now that we've extracted the data we wish to use
data[['LATITUDE_CIRCLE_IMAGE','LONGITUDE_CIRCLE_IMAGE','CENTERED_LATITUDE','CENTERED_LONGITUDE']].describe()

#Because of the bug in seaborn plotting, we now extract the data from the original data frame as arrays and make a new data frame
latitude = numpy.array(data['LATITUDE_CIRCLE_IMAGE'])
diameter = numpy.array(data['DIAM_CIRCLE_IMAGE'])
morphology = numpy.array(data['MORPHOLOGY_EJECTA_1'])
latitudecenter = numpy.array(data['CENTERED_LATITUDE'])
longitude = numpy.array(data['LONGITUDE_CIRCLE_IMAGE'])
depth = numpy.array(data['DEPTH_RIMFLOOR_TOPOG'])
layers = numpy.array(data['NUMBER_LAYERS'])
longitudecenter = numpy.array(data['CENTERED_LONGITUDE'])
data2 = pandas.DataFrame({'LATITUDE':latitude,
                          'DIAMETER':diameter,
                          'MORPHOLOGY_EJECTA_1':morphology,
                          'CENTERED_LATITUDE':latitudecenter,
                          'LONGITUDE':longitude,
                          'CENTERED_LONGITUDE':longitudecenter,
                          'DEPTH':depth,
                          'NUMBER_LAYERS':layers})

print('We now do linear regression for diameter onto latitude for the different ejecta morphologyies.')
seaborn.lmplot(x='CENTERED_LATITUDE',y='DIAMETER',col='MORPHOLOGY_EJECTA_1',hue='MORPHOLOGY_EJECTA_1',data=data2)

#First, let's add in the morphology, longitude, rim depth, and number of layers in our model and see whether
#there are any potential confounding variables
model1 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + CENTERED_LONGITUDE + DEPTH                  + NUMBER_LAYERS + C(MORPHOLOGY_EJECTA_1)',data=data2).fit()
print(model1.summary())

#We notice in this model, that from the p values (<0.05) of the coefficients, that longitude, morphology, and crater 
#depth all show an effect on the final diameter of our crater. Only the number of layers within the crater shows little
#association, so we can safely remove that from our model.

#The surprising association of crater diameter with morphology ejecta type warrants me to separate out craters based on
#their ejecta morphology. Perhaps not too surprising, but the depth of the crater shows the strongest correlation between
#the craters diameter (large coefficient compared to latitude and longitude). We now look at the data one at a time.

model2 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + CENTERED_LONGITUDE',data=data2).fit()
print(model2.summary())
#Here we still that longitude as a potential confounder

model3 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + DEPTH',data=data2).fit()
print(model3.summary())
#Crater is also another potential confounder as th ep value si quite small and we have a large coefficient for depth

model4 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + NUMBER_LAYERS',data=data2).fit()
print(model4.summary())
#Oddly enough, when we only account for the latitude and number of layers within the craters, the number of layers shows a
#negative correlation and is significant. It is only when we add other variables does this disappear.

model5 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + CENTERED_LONGITUDE + DEPTH + NUMBER_LAYERS',data=data2).fit()
print(model5.summary())
#One last check is to remove any categorical variables. When we do this, the number of layers becomes significant again 

#For this analysis, because morphology ejecta may be a potential confounder, we will limit our data to only those craters
#with a morphology of Rd (radial pancake), and look at the latitude and longitude of a given crater, because the original study
#was mostly concerned with relating the diameter of a given crater with its location of impact (of which crater ejecta morphology)
#provided a hint about the underlying terrain which also from the literature has implicit correlation with the crater diameter
#after impact.

data3 = data2.loc[data2['MORPHOLOGY_EJECTA_1'].isin(['Rd'])]

#Because of the bug in seaborn plotting, we now extract the data from the original data frame as arrays and make a new data frame
latitude = numpy.array(data2['LATITUDE'])
diameter = numpy.array(data2['DIAMETER'])
morphology = numpy.array(data2['MORPHOLOGY_EJECTA_1'])
latitudecenter = numpy.array(data2['CENTERED_LATITUDE'])
longitude = numpy.array(data2['LONGITUDE'])
depth = numpy.array(data2['DEPTH'])
layers = numpy.array(data2['NUMBER_LAYERS'])
longitudecenter = numpy.array(data2['CENTERED_LONGITUDE'])
data3 = pandas.DataFrame({'LATITUDE':latitude,
                          'DIAMETER':diameter,
                          'MORPHOLOGY_EJECTA_1':morphology,
                          'CENTERED_LATITUDE':latitudecenter,
                          'LONGITUDE':longitude,
                          'CENTERED_LONGITUDE':longitudecenter,
                          'DEPTH':depth,
                          'NUMBER_LAYERS':layers})

model6 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + CENTERED_LONGITUDE ',data=data3).fit()
print(model6.summary())
model7 = smf.ols(formula = 'DIAMETER ~ CENTERED_LATITUDE + I(CENTERED_LATITUDE**2) + CENTERED_LONGITUDE',data=data3).fit()
print(model7.summary())

#We notice that a warning from Python telling us that we have a large condition number when we try to include a polynomial
#term to our model and also that introducing this term results in no better R-squared value. Therefore we will stick to model6
#for our analysis

#QQ plot for normality
fig1 = sm.qqplot(model6.resid,line='r')


#Plot of the residuals
stdres = pandas.DataFrame(model6.resid_pearson)
plt.plot(stdres, 'o',ls='None')
l = plt.axhline(y=0, color='r')
plt.xlabel('Observation Number')
plt.ylabel('Standardized Residual')

#Additional diagnostics of our regression
fig2 = plt.figure(figsize=[12,8])
fig2 = sm.graphics.plot_regress_exog(model6, "CENTERED_LATITUDE", fig = fig2)

#Our leverage plot
fig3, ax = plt.subplots(figsize=[6,4])
fig3 = sm.graphics.influence_plot(model6,size=8,ax=ax,plot_alpha=1)
ax=fig3.axes[0]
ax.set_xlim(0.0,0.0004)