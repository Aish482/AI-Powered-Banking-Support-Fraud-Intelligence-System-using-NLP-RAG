from app import analyze_query

result = analyze_query(
    query='I see a transaction of ₹10,000 I did not make',
    amount=10000,
    merchant='UnknownMerchant',
    merchant_category='Unknown',
    transaction_type='UPI',
    city='Delhi',
    hour=10,
    day='Monday',
    is_international=False,
    velocity_flag=False,
    geo_flag=False,
    high_amount_flag=False,
)
print('intent=', result['intent'])
print('sentiment=', result['sentiment'])
print('risk_level=', result['risk_level'])
print('fraud_probability=', round(result['fraud_probability'], 3))
print('action=', result['action'])
