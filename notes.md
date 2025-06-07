# Project Notes

## EDA




* Metrics
 MonthlyCharges = TotalCharges / tenure
-> To validate

MonthlyCharges = Services

Many users with low tenure 
-> new users
-> low monthlycharge -> Few products?


Lower tenure -> Higher churn

Higher MonthlyCharges -> Higher churn

Gender -> Similar Churn (male x female)

SeniorCitizen -> Higher Churn

No Partner -> Higher Churn

No Dependents -> Higher Churn

PhoneService -> Similar Churn

MultipleLines -> Higher Churn

InternetService:
Fiber Optic -> Higher Churn
DSL InternetService -> Moderate Churn
No -> Lower churn

OnlineSecurity
No -> Higher Churn
Yes - Moderate Churn
No internet service -> Lower Churn

Online Backup
No -> Higher Churn
Yes -> Moderate Churn
No internet service -> Lower Churn

Device Protection
No -> Higher Churn
Yes -> Moderate Churn
No internet service -> Lower Churn

TechSupport
No -> Higher Churn
Yes -> Moderate Churn
No internet service -> Lower Churn

Streaming TV
No -> Higher Churn
Yes -> Higher Churn
No internet service -> Lower Churn

Streaming Movies
No -> Higher Churn
Yes -> Higher Churn
No internet service -> Lower Churn

Contract Type
Month-to-month -> Higher Churn
One year -> Lower churn (<15%)
Two year -> Lower churn (<5%)

PaperlessBilling
Yes -> Higher Churn
No -> Lower Churn

Payment Method
Electronic check -> Higher Churn
Mail -> Lower Churn
Automatic Transfer -> Lower Churn
Credit Card -> Lower Churn

Hypothesis Test to validate if the difference is stat sig


Avg Product Price = MonthlyCharges / services_count
Total Procuts = services_count


Journey Decision
1. 1st contact
2. Offers
3. Package selection
Family / Partner influence
4. Payment Method
5. Payments
Triggers
Payments problem / service quality / pricing
6. Churn decision
Pricing
Family / Partner influence



Social Factor



Key Findings


Feature Engineering


