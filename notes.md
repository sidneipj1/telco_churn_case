# Project Notes

## EDA

* Metrics
 MonthlyCharges = TotalCharges / tenure 
 Avg Product Price = MonthlyCharges / services_count 
 Total Procuts = services_count
 MonthlyCharges = sum(Services)
 -> To validate



Many users with low tenure 
-> new users
-> low monthlycharge -> Few products?


* Data Findings + Hypothesis

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


* Correlation + Hypothesis test
    Higher tenure -> lower churn
    Higher MonthlyCharges -> higher churn






* Journey Decision
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

* Groups
 - Social
    Dependents + Partner
    SeniorCitizen

 - Payment Friction
    PaymentMethod
    PaperlessBilling

 - Contract
    Contract

 - Products
    amount of products
    Streaming Package
    Online package - internet, security, tech support, backup, protection


 - Pricing
    MonthlyCharges = TotalCharges / tenure 
    Avg Product Price = MonthlyCharges / services_count 
    Total Procuts = services_count
    MonthlyCharges = sum(Services)



* Feature Engineering
    contract_months
        shorter contract -> higher churn
        (inverse)
    Tenure
        higher tenure -> lower churn
        (inverse)
    MonthlyCharge
        higher monthly charge -> higher churn
        (direct)
    MonthlyContract
        Monthly contract -> higher churn
        (direct)
    Streaming_pkg
        Streaming package -> higher churn
        (direct)
    Online_pkg
        Online package -> lower churn
        (inverse)
    Eletronic_check
        Eletronic_check -> higher churn
        (direct)
    Automatic
        Automatic -> lower churn
        (inverse)
    Paperless_billing
        Paperless_billing -> higher churn
        (direct)
    MonthlyCharge <> Charge_tenure_ratio
        Remove Charge_tenure_ratio
        This metric was built to validate MonthlyCharge





