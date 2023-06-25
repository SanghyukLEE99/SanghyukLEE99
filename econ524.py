from flask import Flask, render_template
import pandas as pd
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# FRED API
fred = Fred(api_key='only me')

# FRED
gdp_data = fred.get_series('GDPC1', observation_start='2003-01-01')
unemployment_data = fred.get_series('UNRATE', observation_start='2003-01-01')
population_growth_data = fred.get_series('SPPOPGROWUSA', observation_start='2003-01-01')
technology_data = fred.get_series('PALLFNFINDEXM', observation_start='2003-01-01')
imports_data = fred.get_series('IEAXGS', observation_start='2003-01-01')
exports_data = fred.get_series('IEAMGS', observation_start='2003-01-01')
inflation_data = fred.get_series('FPCPITOTLZGUSA', observation_start='2003-01-01')

# Process missing values
gdp_data = gdp_data.asfreq('MS').fillna(method='ffill')
unemployment_data = unemployment_data.asfreq('MS').fillna(method='ffill')
population_growth_data = population_growth_data.asfreq('MS').fillna(method='ffill')
technology_data = technology_data.asfreq('MS').fillna(method='ffill')
imports_data = imports_data.asfreq('MS').fillna(method='ffill')
exports_data = exports_data.asfreq('MS').fillna(method='ffill')
inflation_data = inflation_data.asfreq('MS').fillna(method='ffill')

# Create DataFrame with independent and dependent variables
data = pd.concat([gdp_data, unemployment_data, population_growth_data, technology_data, imports_data, exports_data, inflation_data], axis=1)
data.columns = ['GDP', 'Unemployment', 'Population Growth', 'Technology', 'Imports', 'Exports', 'Inflation']

# Interpolate missing values using linear interpolation
data = data.interpolate()

# Fit SARIMAX model with exogenous variables
model = sm.tsa.statespace.SARIMAX(data['Inflation'], exog=data.drop(columns=['Inflation']), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast
forecast_start_date = data.index[-1] + pd.DateOffset(months=1) # Start forecast from the next month after the last observation
forecast_end_date = forecast_start_date + pd.DateOffset(months=11)
exog_forecast = data.drop(columns=['Inflation']).iloc[-12:] # Use last 12 months of exogenous data for the forecast
inflation_forecast = results.get_prediction(start=forecast_start_date, end=forecast_end_date, exog=exog_forecast).predicted_mean.rename('Inflation')


# Function to create graph and save as image
def generate_plot():
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(inflation_forecast.index, inflation_forecast, label='Inflation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Inflation')
    ax.set_title('Inflation Forecast')
    ax.legend()

    # Save as Image
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close(fig)

    return plot_url

@app.route('/graphs')
def graphs():
    forecast_plot = generate_plot()

    inflation_forecast_dict = inflation_forecast.to_dict()

    return render_template('graphs.html', plot_url=forecast_plot, inflation_forecast=inflation_forecast_dict)

if __name__ == '__main__':
    try:
        app.run(debug=True, port=8000)

    except SystemExit:
        pass
