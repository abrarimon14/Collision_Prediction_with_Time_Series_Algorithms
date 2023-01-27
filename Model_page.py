from calendar import month
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from dateutil.relativedelta import relativedelta
import time
import datetime


import fbprophet
import pystan
import prophet
from fbprophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from fbprophet.plot import plot_forecast_component_plotly, plot_seasonality_plotly

import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="NYC Traffic Collisions", page_icon=":taxi:", layout="wide")
st.title(":taxi:NYC Traffic Collisions :boom:")
st.markdown("---")
#st.markdown("##")




@st.cache(allow_output_mutation=True)
def get_model(file_name):
    with open(file_name, 'r') as fin:
        return model_from_json(fin.read())  # Load model
model = get_model('serialized_model.json')

@st.cache(allow_output_mutation=True)
def get_master():
    return pd.read_csv("Motor_Vehicle_Collisions_-_Crashes.csv", usecols=['CRASH DATE', 'CRASH TIME', 'BOROUGH'])
master_df = get_master()  #Only get ['CRASH DATE', 'CRASH TIME', 'BOROUGH', 'LATITUDE', 'LONGITUDE']
master_df['CRASH DATE'] = pd.to_datetime(master_df['CRASH DATE'])

@st.cache(allow_output_mutation=True)
def get_mldf():
    return pd.read_csv("mldf_daily.CSV")
mldf = get_mldf()
mldf['ds'] = pd.to_datetime(mldf['ds'])

@st.cache(allow_output_mutation=True)
def get_df_4plot():
    return pd.read_csv("df_4plot_daily.CSV")
df_4plot = get_df_4plot()
df_4plot['ds'] = pd.to_datetime(df_4plot['ds'])

@st.cache(allow_output_mutation=True)
def get_mldf_monthly():
    return pd.read_csv("mldf_monthly.CSV")
mldf_monthly = get_mldf_monthly()
mldf_monthly['ds'] = pd.to_datetime(mldf_monthly['ds'])

@st.cache(allow_output_mutation=True)
def get_monthly_df_4plot():
    return pd.read_csv("df_4plot_monthly.CSV")
monthly_df_4plot = get_monthly_df_4plot()
monthly_df_4plot['ds'] = pd.to_datetime(monthly_df_4plot['ds'])

@st.cache(allow_output_mutation=True)
def get_bor(path):
    temporary = pd.read_csv(path)
    temporary['ds'] = pd.to_datetime(temporary['ds'])
    return temporary

# This loop will read in the individual predictions for the boroughs
for bor in ['bronx', 'brooklyn', 'manhattan', 'queens', 'staten']:
    for name in ['_4plot_daily', '_4plot_monthly', '_mldf_daily', "_mldf_monthly"]:
        #FOR ELIAN: since you will save the data in a directory called "data", change next line to p = 'data\\'+bor+name+'.CSV'
        #... If all of the borough data are in a different directory, fill that path in with double slashes like this: p = 'C:\\Users\\shipa\\...data\\' + bor + name + '.CSV'
        p = bor + name + '.CSV'
        exec(bor+name+"= get_bor(p)")  #exec is used as a way to set variables in which the var names come from strings. Here: bor and name are actually strings.
        
#FOR ELIAN: To see if the above loop worked, uncomment and run the next line and see if you get a dataframe. Remember to comment it in again
#manhattan_4plot_daily.head()

# def predict_daily(d):
#     ser = pd.DataFrame(np.array(np.array([d]), dtype='datetime64[ns]'), columns = ['ds'])
#     ser['on_school'] = ser['ds'].apply(is_school_season)
#     ser['off_school'] = ~ser['ds'].apply(is_school_season)
#     return model.predict(ser).loc[0,['yhat', 'yhat_lower', 'yhat_upper']].clip(lower=0.0)

#predict_daily('2023-01-01')

#region searchbar

st.subheader("Query Time Period")

# date_selected = st.text_input("Specify Year, Month, or Day (ex. 2022-01-31):", "2022")  #note: takes input as str
time_form = st.date_input("Specify Date:", datetime.date(2022, 1, 1), min_value=datetime.date(2012, 7, 1), max_value=datetime.date(2023, 12, 21))

def bor_count(ddf):
    return ddf[ddf['ds']==time_form]['y'].iloc[0]

left1, right1 = st.columns([1,1.5])

if time_form:
    time_form = datetime.datetime.strptime(str(time_form), "%Y-%m-%d")
    with left1:
        if time_form >= df_4plot['ds'].min():
            st.markdown('<p style="font-family:sans-serif; color:rgb(231,107,243); font-size: 14px; text-align: left">{}</p>'.format(
                'Expected Number of Collisions on '+ str(time_form)[:-9] +':'),
                unsafe_allow_html=True)
            st.markdown('<p style="font-family:sans-serif; color:rgb(231,107,243); font-size: 30px; text-align: left">{}</p>'.format(
                str(df_4plot[df_4plot['ds']==time_form]['y'].iloc[0])[:-2]),
                unsafe_allow_html=True)
            st.markdown('<p style="font-family:sans-serif; color:rgb(231,107,243); font-size: 14px; text-align: left">Uncertainty Interval: ({lower}, {upper})</p>'.format(
                lower = str(df_4plot[df_4plot['ds']==time_form]['yhat_lower'].iloc[0])[:-2],
                upper = str(df_4plot[df_4plot['ds']==time_form]['yhat_upper'].iloc[0])[:-2]),
                unsafe_allow_html=True)
        else:
            if time_form >= mldf['ds'].min():
                st.markdown('<p style="font-family:sans-serif; color:rgb(0,176,246); font-size: 14px; text-align: left">{}</p>'.format('Number of Collisions on '+ str(time_form)[:-9] +':'),
                    unsafe_allow_html=True)
                st.markdown('<p style="font-family:sans-serif; color:rgb(0,176,246); font-size: 30px; text-align: left">{}</p>'.format(
                    mldf[mldf['ds']==time_form]['y'].iloc[0]),
                    unsafe_allow_html=True)
                st.text("")

    with right1:
        if time_form >= df_4plot['ds'].min():
            st.markdown('<p style="font-family:sans-serif; color:rgb(231,107,243); font-size: 14px; text-align: center">{}</p>'.format('Predicted Percentage of Collisions by Borough'),
                    unsafe_allow_html=True)
            z = pd.DataFrame(data={'Borough':['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
                    'count':[bor_count(bronx_4plot_daily),
                            bor_count(brooklyn_4plot_daily),
                            bor_count(manhattan_4plot_daily),
                            bor_count(queens_4plot_daily),
                            bor_count(staten_4plot_daily)]}
                )
            pie = px.pie(z, values = 'count',
                names='Borough',
                #title='Count of Collisions by Borough',
                color_discrete_sequence=px.colors.sequential.Magenta
                )
            #pie.update_traces(textinfo='value')
            st.plotly_chart(pie, use_container_width=True)
            
        
        else:
            st.markdown('<p style="font-family:sans-serif; color:rgb(0,176,246); font-size: 14px; text-align: center">{}</p>'.format('Percentage of Collisions by Borough'),
                    unsafe_allow_html=True)
            # z = master_df[master_df['CRASH DATE'] ==  time_form].groupby(['BOROUGH'])['BOROUGH'].count().to_frame()
            # z.columns = ['count']
            # z=z.reset_index()
            z = pd.DataFrame(data={'Borough':['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'],
                    'count':[bor_count(bronx_mldf_daily),
                            bor_count(brooklyn_mldf_daily),
                            bor_count(manhattan_mldf_daily),
                            bor_count(queens_mldf_daily),
                            bor_count(staten_mldf_daily)]}
                )
            pie = px.pie(z, values = 'count',
                names='Borough',
                #title='Count of Collisions by Borough',
                color_discrete_sequence=px.colors.sequential.Aggrnyl
                )
            #pie.update_traces(textinfo='value')
            st.plotly_chart(pie, use_container_width=True)


#endregion searchbar


#region Linechart
st.markdown("---")

st.subheader("Line Chart")


#st.dataframe(monthly_df_4plot.head())
time_selected = st.selectbox("Aggregate by:", ["Daily", "Monthly"])


#fig = px.line(df_4plot, x="ds", y="y")

# day_checked = st.checkbox("Aggregate by Day", value=True)
# month_checked = st.checkbox("Aggregate by Month")


if time_selected == "Daily":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mldf['ds'], y=mldf['y'],
                        mode='lines',
                        line_color='rgb(0,176,246)',
                        name='Historical'))
    fig.update_xaxes(title_text = "Date")
    fig.update_yaxes(title_text = "Number of Accidents per Day")
    fig.add_trace(go.Scatter(x=df_4plot['ds'], y=df_4plot['y'],
                        mode='lines',
                        line_color='rgb(231,107,243)',
                        name='Future'))
    x_rev = list(df_4plot['ds'][::-1])
    fig.add_trace(go.Scatter(x=list(df_4plot['ds'])+x_rev, y=list(df_4plot['yhat_upper'])+list(df_4plot['yhat_lower'][::-1]),
                        fill='toself',
                        fillcolor='rgba(231,107,243,0.2)',
                        line_color='rgba(255,255,255,0)',
                        mode='lines',
                        name='Prediction Interval'))
    fig.update_layout(yaxis_range=[0,700])
    fig.update_layout(xaxis_range=[mldf.iloc[-1, 0] - relativedelta(weeks=15), mldf.iloc[-1, 0] + relativedelta(weeks=3)])
    fig.update_layout(width=1000, height=500)

    st.plotly_chart(fig, use_container_width=True)

elif time_selected == "Monthly":
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mldf_monthly['ds'], y=mldf_monthly['y'],
                        mode='lines',
                        line_color='rgb(0,176,246)',
                        name='Historical'))
    fig.update_xaxes(title_text = "Date")
    fig.update_yaxes(title_text = "Number of Accidents per Month")
    fig.add_trace(go.Scatter(x=monthly_df_4plot['ds'], y=monthly_df_4plot['y'],
                        mode='lines',
                        line_color='rgb(231,107,243)',
                        name='Future'))
    x_rev = list(monthly_df_4plot['ds'][::-1])
    fig.add_trace(go.Scatter(x=list(monthly_df_4plot['ds'])+x_rev, y=list(monthly_df_4plot['yhat_upper'])+list(monthly_df_4plot['yhat_lower'][::-1]),
                        fill='toself',
                        fillcolor='rgba(231,107,243,0.2)',
                        line_color='rgba(255,255,255,0)',
                        mode='lines',
                        name='Prediction Interval'))
    #fig.update_layout(yaxis_range=[0,15000])
    #fig.update_layout(xaxis_range=[mldf_monthly.iloc[-1, 0] - relativedelta(months=36), mldf_monthly.iloc[-1, 0] + relativedelta(months=12)])
    fig.update_layout(width=1000, height=500)

    st.plotly_chart(fig, use_container_width=True)

# list(monthly_df_4plot['yhat_lower'])
# monthly_df_4plot
# monthly_df_4plot['ds'][::-1]


#endregion Linechart


st.markdown("---")

st.subheader("Patterns")



patt_selected = st.selectbox("Select Pattern type:", ["Day of the Week", "Day of the Year", "Holidays"])

left3, mid3,  right3 = st.columns([.25, 1, .25])




with mid3:
    if patt_selected == "Day of the Week":
        st.write("The first plot is for when school is in session. The second plot is for when school is not in session.")
        st.plotly_chart(plot_seasonality_plotly(model, "weekly_on_season"))
        st.plotly_chart(plot_seasonality_plotly(model, "weekly_off_season"))

    elif patt_selected == "Day of the Year":
        st.plotly_chart(plot_seasonality_plotly(model, "yearly"), use_container_width=True)

    elif patt_selected == "Holidays":
        st.write("Note: these effects are multiplied to the yearly seasonality at the time of the holidays.")
        def is_school_season(ds):
            date = pd.to_datetime(ds)
            return (date.month > 8 or date.month < 7)

        future_dates = model.make_future_dataframe(periods=730)
        future_dates['on_school'] = future_dates['ds'].apply(is_school_season)
        future_dates['off_school'] = ~future_dates['ds'].apply(is_school_season)
        prediction=model.predict(future_dates)

        st.plotly_chart(plot_forecast_component_plotly(model, prediction, "holidays"), use_container_width=True)

    # elif patt_selected == "Errors":
    #     st.write("These errors came from cross validation on the time series model.")
    #     st.image(error_image, caption='Mean Absolute Error')






# if patt_selected == "Day of the Week":
#     st.image(weekly_image, caption='Weekly Seasonality' )












