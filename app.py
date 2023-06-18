import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template

# 예시 데이터와 generate_plot() 함수
inflation_forecast = pd.DataFrame({"Date": ["2023-04", "2023-05", "2023-06"],
                                   "Forecasted Inflation": [2.0, 2.1, 2.2]})

def generate_plot():
    fig, ax = plt.subplots()
    ax.plot(inflation_forecast["Date"], inflation_forecast["Forecasted Inflation"])
    ax.set_ylabel("Inflation (%)")
    ax.set_title("Inflation Forecast")

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    return plot_url

app = Flask(__name__)

def graphs():
    forecast_plot = generate_plot()
    inflation_forecast_dict = inflation_forecast.to_dict()
    return forecast_plot, inflation_forecast_dict

@app.route('/')
def index():
    forecast_plot, inflation_forecast_dict = graphs()
    return render_template('index.html', plot_url=forecast_plot, inflation_forecast=inflation_forecast_dict)

if __name__ == '__main__':
    app.run(debug=True)
