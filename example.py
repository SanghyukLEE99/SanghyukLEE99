from flask import Flask, render_template
import pandas as pd
from fredapi import Fred
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# FRED API 키 설정
fred = Fred(api_key='5fa5ff016239309549ecd0184a37535e')

# FRED 데이터 다운로드
inflation_data = fred.get_series('FPCPITOTLZGUSA', observation_start='1973-01-01')
awi_data = fred.get_series('AHETPI', observation_start='1973-01-01')
psr_data = fred.get_series('PSAVERT', observation_start='1973-01-01')

# 결측값 처리
inflation_data = inflation_data.fillna(method='ffill')
awi_data = awi_data.fillna(method='ffill')
psr_data = psr_data.fillna(method='ffill')

# SARIMA 모델로 예측하기
inflation_model = sm.tsa.statespace.SARIMAX(inflation_data, order=(1,1,1), seasonal_order=(0,1,1,12))
awi_model = sm.tsa.statespace.SARIMAX(awi_data, order=(1,1,1), seasonal_order=(0,1,1,12))
psr_model = sm.tsa.statespace.SARIMAX(psr_data, order=(1,1,1), seasonal_order=(0,1,1,12))

inflation_results = inflation_model.fit()
awi_results = awi_model.fit()
psr_results = psr_model.fit()

# 예측치 생성
inflation_forecast = inflation_results.forecast(steps=12)
awi_forecast = awi_results.forecast(steps=12)
psr_forecast = psr_results.forecast(steps=12)

# 데이터프레임 생성
forecast_df = pd.DataFrame({'inflation': inflation_forecast,
                            'average_hourly_earnings': awi_forecast,
                            'personal_savings_rate': psr_forecast},
                            index=pd.date_range(start='2022-01-01', periods=12, freq='M'))


def generate_plot():
    # 그래프 생성
    plt.figure()
    plt.plot(forecast_df.index, forecast_df['inflation'], label='Inflation')
    plt.plot(forecast_df.index, forecast_df['average_hourly_earnings'], label='Average Hourly Earnings')
    plt.plot(forecast_df.index, forecast_df['personal_savings_rate'], label='Personal Savings Rate')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Forecasted Values')
    plt.legend()

    # 이미지로 저장
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return plot_url


@app.route('/')
def index():
    # 그래프 생성 및 이미지 URL 생성
    plot_url = generate_plot()

    return render_template('index.html', tables=[forecast_df.to_html(classes='data', header='true')], plot_url=plot_url)


if __name__ == '__main__':
    app.run(debug=True)
