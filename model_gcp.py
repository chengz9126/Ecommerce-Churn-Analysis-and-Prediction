#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# load the trained model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("trained_model.model")

@app.route('/predict', methods=['POST'])
def predict():
    # get the input data
    data = request.get_json()
    tenure = data['Tenure']
    preferred_login_device = data['PreferredLoginDevice']
    city_tier = data['CityTier']
    warehouse_to_home = data['WarehouseToHome']
    preferred_payment_mode = data['PreferredPaymentMode']
    gender = data['Gender']
    hours_spend_on_app = data['HourSpendOnApp']
    number_of_device_registered = data['NumberOfDeviceRegistered']
    prefered_order_cat = data['PreferedOrderCat']
    satisfaction_score = data['SatisfactionScore']
    marital_status = data['MaritalStatus']
    number_of_address = data['NumberOfAddress']
    complain = data['Complain']
    order_amount_hike_from_last_year = data['OrderAmountHikeFromlastYear']
    coupon_used = data['CouponUsed']
    order_count = data['OrderCount']
    day_since_last_order = data['DaySinceLastOrder']
    cashback_amount = data['CashbackAmount']

    # create input data
    input_data = np.array([tenure,preferred_login_device,city_tier,warehouse_to_home,preferred_payment_mode,gender,hours_spend_on_app,number_of_device_registered,prefered_order_cat,satisfaction_score,marital_status,number_of_address,complain,order_amount_hike_from_last_year,coupon_used,order_count,day_since_last_order,cashback_amount]).reshape(1, -1)

    # make predictions
    prediction = xgb_model.predict(input_data)

    # return the result
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

