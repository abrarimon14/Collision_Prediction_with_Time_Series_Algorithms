# Collision_Prediction_with_Time_Series_Algorithms
This repo contains only my contributions to the repo, "CSCI-233-group-project". The main group project includes more pages, which focus on exploratory analysis of data from NYC Open Data that have to do with traffic collisions. My contribution was creating models that predict future numbers of collisions in different boroughs of NYC. Then, I created the page of the website that relates to my models.

Descriptions of the files:

model_page_demo.mp4 - A video that showcases my page of the website that is about time series model.

ML_script.py -	Trained multiplicative time series models to predict number of collisions in the future in different boroughs. The models account for the effects of day of week and year, and many American holidays on the number of collisions. The model uses FbProphet changepoint detection to find a trendline. The model also accounts for the effects of whether school is open or not, on the weekly seasonality.

Model_dashboard.py - Created a dashboard that show the predictions with interactive plots. Created search bar to query specific dates. Created a section that allow users to understand the interpretable patterns discovered by the fourier transform.

