from calendar import month
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dateutil.relativedelta import relativedelta

import fbprophet
import pystan
import prophet
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json


master_df = pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv", usecols=['CRASH DATE', 'CRASH TIME', 'BOROUGH'])


#region Daily
df_date = master_df['CRASH DATE'].to_frame()
df_date['CRASH DATE'] = pd.to_datetime(df_date['CRASH DATE'])
df_date = df_date['CRASH DATE'].dt.date.value_counts().sort_index().reset_index()
df_date.columns = ['Date', 'Count']

mldf = df_date.copy()
mldf.columns = ['ds', 'y']
mldf['ds'] = pd.to_datetime(mldf['ds'])

def is_school_season(ds):
    date = pd.to_datetime(ds)
    return (date.month > 8 or date.month < 7)

mldf['on_school'] = mldf['ds'].apply(is_school_season)
mldf['off_school'] = ~mldf['ds'].apply(is_school_season)

covid = mldf.loc[(mldf['ds'] >= '2019-12-01') & (mldf['ds'] <= '2020-09-01')]

pre_covid = mldf.loc[(mldf['ds'] < '2020-02-01')].copy()
post_covid = mldf.loc[(mldf['ds'] >= '2020-05-01')].copy()

diff = round(pre_covid['y'].mean() - post_covid['y'].mean())
precov_modified = pre_covid.copy()
#Stretch for multiplicative features
precov_modified['y'] = (precov_modified['y']*post_covid['y'].mean())/(pre_covid['y'].mean())

mldf_nocov = pd.concat([precov_modified, post_covid])

model=Prophet(changepoint_prior_scale=0.047,
    changepoint_range=0.94,
    weekly_seasonality=False,
    #holidays_prior_scale=6,
    seasonality_mode='multiplicative'
    )
model.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_school')
model.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_school')

model.add_country_holidays(country_name='US')
model.fit(mldf_nocov)


future_dates = model.make_future_dataframe(periods=730)
future_dates['on_school'] = future_dates['ds'].apply(is_school_season)
future_dates['off_school'] = ~future_dates['ds'].apply(is_school_season)
prediction=model.predict(future_dates)


for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    prediction[col] = prediction[col].clip(lower=0.0)

dfpred = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].loc[prediction['ds'] > mldf['ds'].max()]
dfpred.columns = ['ds', 'y', 'yhat_lower', 'yhat_upper']

df_4plot = dfpred.copy()
df_4plot = pd.concat([mldf.iloc[[-1]], df_4plot])
df_4plot['yhat_lower'].iloc[0] = df_4plot['y'].iloc[0]
df_4plot['yhat_upper'].iloc[0] = df_4plot['y'].iloc[0]

mldf = mldf.drop(['on_school', 'off_school'], axis=1)

df_4plot = df_4plot.drop(['on_school', 'off_school'], axis=1)
df_4plot['y'] = round(df_4plot['y'])
df_4plot['yhat_lower'] = round(df_4plot['yhat_lower'])
df_4plot['yhat_upper'] = round(df_4plot['yhat_upper'])


with open('serialized_model.json', 'w') as fout:
    fout.write(model_to_json(model))  # Save model

mldf.to_csv('mldf_daily.CSV', index=False)
df_4plot.to_csv('df_4plot_daily.CSV', index=False)

#endregion Daily


#region monthly
mldf_monthly = master_df[['CRASH DATE']]
mldf_monthly['date'] = pd.to_datetime(mldf_monthly["CRASH DATE"])
mldf_monthly=mldf_monthly.drop(["CRASH DATE"], axis=1)
mldf_monthly['month']=mldf_monthly['date'].dt.month
mldf_monthly['year']=mldf_monthly['date'].dt.year
mldf_monthly=mldf_monthly.groupby(['year','month']).count()
mldf_monthly=mldf_monthly.reset_index()

mldf_monthly['new'] = mldf_monthly['year'].astype(str)+'-'+mldf_monthly['month'].astype(str)
mldf_monthly['newdate']=pd.to_datetime(mldf_monthly["new"])
mldf_monthly=mldf_monthly.drop(['year','month'], axis=1)
mldf_monthly.columns=['count', 'month(str)','month']
mldf_monthly=mldf_monthly.drop('month(str)', axis=1)
mldf_monthly = mldf_monthly[['month', 'count']]
mldf_monthly.columns = ['ds', 'y']
mldf_monthly = mldf_monthly[:-1]


monthly_pre_covid = mldf_monthly.loc[(mldf_monthly['ds'] < '2020-02-01')].copy()
monthly_post_covid = mldf_monthly.loc[(mldf_monthly['ds'] >= '2020-05-01')].copy()

covid_diff_monthly = round(monthly_pre_covid['y'].mean() - monthly_post_covid['y'].mean())
monthly_precov_modified = monthly_pre_covid.copy()
monthly_precov_modified['y'] = ((monthly_precov_modified['y'])*(monthly_post_covid['y'].mean()))/(monthly_pre_covid['y'].mean())

monthly_mldf_nocov = pd.concat([monthly_precov_modified, monthly_post_covid])

model_monthly=Prophet(changepoint_prior_scale=0.049,
    changepoint_range=0.89,
    weekly_seasonality=False,
    #holidays_prior_scale=6,
    seasonality_mode='multiplicative'
    )
model_monthly.fit(monthly_mldf_nocov)

monthly_future_dates = model_monthly.make_future_dataframe(periods=24, freq='MS')

monthly_prediction=model_monthly.predict(monthly_future_dates)

for col in ['yhat', 'yhat_lower', 'yhat_upper']:
    monthly_prediction[col] = monthly_prediction[col].clip(lower=0.0)



#This df ONLY has the days after today
monthly_dfpred = monthly_prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].loc[monthly_prediction['ds'] > mldf_monthly['ds'].max()]
monthly_dfpred.columns = ['ds', 'y', 'yhat_lower', 'yhat_upper']


monthly_df_4plot = monthly_dfpred.copy()
monthly_df_4plot = pd.concat([mldf_monthly.iloc[[-1]], monthly_df_4plot]).copy()
monthly_df_4plot['yhat_lower'].iloc[0] = monthly_df_4plot['y'].iloc[0]
monthly_df_4plot['yhat_upper'].iloc[0] = monthly_df_4plot['y'].iloc[0]

# bor_df_4plot['yhat_upper'] = round(bor_df_4plot['yhat_upper'])
monthly_df_4plot['y'] = round(monthly_df_4plot['y'])
monthly_df_4plot['yhat_lower'] = round(monthly_df_4plot['yhat_lower'])
monthly_df_4plot['yhat_upper'] = round(monthly_df_4plot['yhat_upper'])

mldf_monthly.to_csv('mldf_monthly.CSV', index=False)
monthly_df_4plot.to_csv('df_4plot_monthly.CSV', index=False)

#endregion


#region borough daily


bor_off = master_df[master_df['BOROUGH'].isna()]
null_df_date = bor_off['CRASH DATE'].to_frame()
null_df_date['CRASH DATE'] = pd.to_datetime(null_df_date['CRASH DATE'])
null_df_date = null_df_date['CRASH DATE'].dt.date.value_counts().sort_index().reset_index()
null_df_date.columns = ['Date', 'null_count']

bor_only = master_df[master_df['BOROUGH'].notna()]

def daily_pred_borough(borough_name, filename1, filename2):  #take input as string just like value name in csv
    borough_df = master_df[master_df['BOROUGH']==borough_name]
    prop = len(borough_df)/len(bor_only)

    bor_df_date = borough_df['CRASH DATE'].to_frame()
    bor_df_date['CRASH DATE'] = pd.to_datetime(bor_df_date['CRASH DATE'])
    bor_df_date = bor_df_date['CRASH DATE'].dt.date.value_counts().sort_index().reset_index()
    bor_df_date.columns = ['Date', 'Count']

    bor_df_date = bor_df_date.merge(null_df_date, how='left', on='Date')
    bor_df_date['Count2'] = bor_df_date['Count'] + round((bor_df_date['null_count'])*(prop))
    bor_df_date = bor_df_date.drop(['null_count', 'Count'], axis=1)
    bor_df_date.columns = ['Date', 'Count']

    bor_mldf = bor_df_date.copy()
    bor_mldf.columns = ['ds', 'y']
    bor_mldf['ds'] = pd.to_datetime(bor_mldf['ds'])

    bor_mldf['on_school'] = bor_mldf['ds'].apply(is_school_season)
    bor_mldf['off_school'] = ~bor_mldf['ds'].apply(is_school_season)

    covid = bor_mldf.loc[(bor_mldf['ds'] >= '2019-12-01') & (bor_mldf['ds'] <= '2020-09-01')]

    pre_covid = bor_mldf.loc[(bor_mldf['ds'] < '2020-02-01')].copy()
    post_covid = bor_mldf.loc[(bor_mldf['ds'] >= '2020-05-01')].copy()

    diff = round(pre_covid['y'].mean() - post_covid['y'].mean())
    precov_modified = pre_covid.copy()
    #Stretch for multiplicative features
    precov_modified['y'] = (precov_modified['y']*post_covid['y'].mean())/(pre_covid['y'].mean())

    bor_mldf_nocov = pd.concat([precov_modified, post_covid])

    bor_model=Prophet(changepoint_prior_scale=0.047,
    changepoint_range=0.94,
    weekly_seasonality=False,
    #holidays_prior_scale=6,
    seasonality_mode='multiplicative'
    )
    bor_model.add_seasonality(name='weekly_on_season', period=7, fourier_order=3, condition_name='on_school')
    bor_model.add_seasonality(name='weekly_off_season', period=7, fourier_order=3, condition_name='off_school')

    bor_model.add_country_holidays(country_name='US')
    bor_model.fit(bor_mldf_nocov)

    future_dates = bor_model.make_future_dataframe(periods=730)
    future_dates['on_school'] = future_dates['ds'].apply(is_school_season)
    future_dates['off_school'] = ~future_dates['ds'].apply(is_school_season)
    prediction=bor_model.predict(future_dates)

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        prediction[col] = prediction[col].clip(lower=0.0)

    dfpred = prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].loc[prediction['ds'] > bor_mldf['ds'].max()]
    dfpred.columns = ['ds', 'y', 'yhat_lower', 'yhat_upper']

    bor_df_4plot = dfpred.copy()
    bor_df_4plot = pd.concat([bor_mldf.iloc[[-1]], bor_df_4plot])
    bor_df_4plot['yhat_lower'].iloc[0] = bor_df_4plot['y'].iloc[0]
    bor_df_4plot['yhat_upper'].iloc[0] = bor_df_4plot['y'].iloc[0]

    bor_mldf = bor_mldf.drop(['on_school', 'off_school'], axis=1)

    bor_df_4plot = bor_df_4plot.drop(['on_school', 'off_school'], axis=1)
    bor_df_4plot['y'] = round(bor_df_4plot['y'])
    bor_df_4plot['yhat_lower'] = round(bor_df_4plot['yhat_lower'])
    bor_df_4plot['yhat_upper'] = round(bor_df_4plot['yhat_upper'])


    bor_mldf.to_csv(filename1+'.CSV', index=False)
    bor_df_4plot.to_csv(filename2+'.CSV', index=False)


daily_pred_borough('BROOKLYN', 'brooklyn_mldf_daily', 'brooklyn_4plot_daily')
daily_pred_borough('BRONX', 'bronx_mldf_daily', 'bronx_4plot_daily')
daily_pred_borough('MANHATTAN', 'manhattan_mldf_daily', 'manhattan_4plot_daily')
daily_pred_borough('QUEENS', 'queens_mldf_daily', 'queens_4plot_daily')
daily_pred_borough('STATEN ISLAND', 'staten_mldf_daily', 'staten_4plot_daily')

#endregion


#region borough monthly
#note bor_off is masters with only null, bor_only has not nulls
null_df_month = bor_off['CRASH DATE'].to_frame()
null_df_month['date'] = pd.to_datetime(null_df_month["CRASH DATE"])
null_df_month=null_df_month.drop(["CRASH DATE"], axis=1)
null_df_month['month']=null_df_month['date'].dt.month
null_df_month['year']=null_df_month['date'].dt.year
null_df_month=null_df_month.groupby(['year','month']).count()
null_df_month=null_df_month.reset_index()

null_df_month['new'] = null_df_month['year'].astype(str)+'-'+null_df_month['month'].astype(str)
null_df_month['newdate']=pd.to_datetime(null_df_month["new"])
null_df_month=null_df_month.drop(['year','month'], axis=1)
null_df_month.columns=['count', 'month(str)','month']
null_df_month=null_df_month.drop('month(str)', axis=1)
null_df_month = null_df_month[['month', 'count']]
null_df_month.columns = ['ds', 'y']
null_df_month = null_df_month[:-1]

def monthly_pred_borough(borough_name, filename1, filename2):  #take input as string just like value name in csv
    borough_df = master_df[master_df['BOROUGH']==borough_name]
    prop = len(borough_df)/len(bor_only)

    bor_mldf_monthly = borough_df[['CRASH DATE']]
    bor_mldf_monthly['date'] = pd.to_datetime(bor_mldf_monthly["CRASH DATE"])
    bor_mldf_monthly=bor_mldf_monthly.drop(["CRASH DATE"], axis=1)
    bor_mldf_monthly['month']=bor_mldf_monthly['date'].dt.month
    bor_mldf_monthly['year']=bor_mldf_monthly['date'].dt.year
    bor_mldf_monthly=bor_mldf_monthly.groupby(['year','month']).count()
    bor_mldf_monthly=bor_mldf_monthly.reset_index()

    bor_mldf_monthly['new'] = bor_mldf_monthly['year'].astype(str)+'-'+bor_mldf_monthly['month'].astype(str)
    bor_mldf_monthly['newdate']=pd.to_datetime(bor_mldf_monthly["new"])
    bor_mldf_monthly=bor_mldf_monthly.drop(['year','month'], axis=1)
    bor_mldf_monthly.columns=['count', 'month(str)','month']
    bor_mldf_monthly=bor_mldf_monthly.drop('month(str)', axis=1)
    bor_mldf_monthly = bor_mldf_monthly[['month', 'count']]
    bor_mldf_monthly.columns = ['ds', 'y1']
    bor_mldf_monthly = bor_mldf_monthly[:-1]


    bor_mldf_monthly = bor_mldf_monthly.merge(null_df_month, how='left', on='ds')
    bor_mldf_monthly['y3'] = bor_mldf_monthly['y1'] + round((bor_mldf_monthly['y'])*(prop))
    bor_mldf_monthly = bor_mldf_monthly.drop(['y', 'y1'], axis=1)
    bor_mldf_monthly.columns = ['ds', 'y']

    bor_mldf_monthly['ds'] = pd.to_datetime(bor_mldf_monthly['ds'])

    bor_monthly_pre_covid = bor_mldf_monthly.loc[(bor_mldf_monthly['ds'] < '2020-02-01')].copy()
    bor_monthly_post_covid = bor_mldf_monthly.loc[(bor_mldf_monthly['ds'] >= '2020-05-01')].copy()

    bor_monthly_precov_modified = bor_monthly_pre_covid.copy()
    bor_monthly_precov_modified['y'] = ((bor_monthly_precov_modified['y'])*(bor_monthly_post_covid['y'].mean()))/(bor_monthly_pre_covid['y'].mean())

    bor_monthly_mldf_nocov = pd.concat([bor_monthly_precov_modified, bor_monthly_post_covid])

    bor_model_monthly=Prophet(changepoint_prior_scale=0.049,
        changepoint_range=0.89,
        weekly_seasonality=False,
        #holidays_prior_scale=6,
        seasonality_mode='multiplicative'
        )
    bor_model_monthly.fit(bor_monthly_mldf_nocov)

    bor_monthly_future_dates = bor_model_monthly.make_future_dataframe(periods=24, freq='MS')

    bor_monthly_prediction=bor_model_monthly.predict(bor_monthly_future_dates)

    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        bor_monthly_prediction[col] = bor_monthly_prediction[col].clip(lower=0.0)

    #This df ONLY has the days after today
    bor_monthly_dfpred = bor_monthly_prediction[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].loc[bor_monthly_prediction['ds'] > bor_mldf_monthly['ds'].max()]
    bor_monthly_dfpred.columns = ['ds', 'y', 'yhat_lower', 'yhat_upper']

    bor_monthly_df_4plot = bor_monthly_dfpred.copy()
    bor_monthly_df_4plot = pd.concat([bor_mldf_monthly.iloc[[-1]], bor_monthly_df_4plot]).copy()
    bor_monthly_df_4plot['yhat_lower'].iloc[0] = bor_monthly_df_4plot['y'].iloc[0]
    bor_monthly_df_4plot['yhat_upper'].iloc[0] = bor_monthly_df_4plot['y'].iloc[0]

    bor_monthly_df_4plot['y'] = round(bor_monthly_df_4plot['y'])
    bor_monthly_df_4plot['yhat_lower'] = round(bor_monthly_df_4plot['yhat_lower'])
    bor_monthly_df_4plot['yhat_upper'] = round(bor_monthly_df_4plot['yhat_upper'])

    bor_mldf_monthly.to_csv(filename1+'.CSV', index=False)
    bor_monthly_df_4plot.to_csv(filename2+'.CSV', index=False)



monthly_pred_borough('BROOKLYN', 'brooklyn_mldf_monthly', 'brooklyn_4plot_monthly')
monthly_pred_borough('BRONX', 'bronx_mldf_monthly', 'bronx_4plot_monthly')
monthly_pred_borough("MANHATTAN", "manhattan_mldf_monthly", "manhattan_4plot_monthly")
monthly_pred_borough('QUEENS', 'queens_mldf_monthly', 'queens_4plot_monthly')
monthly_pred_borough('STATEN ISLAND', 'staten_mldf_monthly', 'staten_4plot_monthly')


#endregion










