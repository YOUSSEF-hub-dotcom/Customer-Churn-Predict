import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Initialize logger for tracking exploratory data analysis stages
logger = logging.getLogger("EDA_1")

def basic_eda(df):
    """
    Performs comprehensive Univariate Exploratory Data Analysis (EDA) 
    and generates descriptive visualizations for all key features.
    """
    logger.info("============ Exploratory Data Analysis (Basic) & Visualization ============")

    # -------------------------------------------------------------------------
    # 1. Gender Distribution
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of gender in the dataset?")
    dist_gender = df['gender'].value_counts()
    print(dist_gender)
    logger.info("Distribution of Female: 49.6% , Male: 50.4%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='gender')
    plt.title('Distribution of Gender')
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Senior Citizen Distribution (Mapped to 0/1 in preprocessing)
    # -------------------------------------------------------------------------
    logger.info("How many customers are Senior Citizens?")
    num_senior_citizens = df['SeniorCitizen'].value_counts()
    print(num_senior_citizens)
    logger.info("Percentage of customers Senior Citizens (elderly people) = 16.2%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='SeniorCitizen')
    plt.title('Distribution of Senior Citizens')
    plt.xticks([0, 1], ['Not Senior Citizen (0)', 'Senior Citizen (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 3. Partner Status Distribution (Mapped to 0/1)
    # -------------------------------------------------------------------------
    logger.info("How many customers have partners?")
    num_partners = df['Partner'].value_counts()
    print(num_partners)
    logger.info("Percentage of customers Partners = 48.3%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Partner')
    plt.title('Customers with Partner')
    plt.xticks([0, 1], ['Not Partners (0)', 'Partners (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 4. Dependents Distribution (Mapped to 0/1)
    # -------------------------------------------------------------------------
    logger.info("How many customers have dependents?")
    num_dependents = df['Dependents'].value_counts()
    print(num_dependents)
    logger.info("Percentage of customers Dependents = 29.9%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Dependents')
    plt.title('Customers with Dependents')
    plt.xticks([0, 1], ['Not Dependents (0)', 'Dependents (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 5. Customer Tenure Distribution (Continuous Numerical)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of tenure?")
    num_tenure = df['tenure'].describe()
    print(num_tenure)
    logger.info("The tenure of customers ranges from 0 to 72 months, "
                "with a mean of 32 months. Most customers have a tenure around 2-3 years. "
                "New customers (tenure <= 12 months) are more likely to churn, "
                "while long-term customers (tenure >= 49 months) tend to be loyal.")

    plt.figure(figsize=(8, 4))
    sns.indigo = sns.histplot(data=df, x='tenure', bins=30, kde=True)
    plt.title('Distribution of Tenure')
    plt.show()

    # -------------------------------------------------------------------------
    # 6. Phone Service Distribution (Mapped to 0/1)
    # -------------------------------------------------------------------------
    logger.info("How many customers have Phone Service?")
    num_phone_service = df['PhoneService'].value_counts()
    print(num_phone_service)
    logger.info("Percentage of customers there have Phone Services = 90.3%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='PhoneService')
    plt.title('Customers with Phone Service')
    plt.xticks([0, 1], ['Not PhoneService (0)', 'PhoneService (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 7. Multiple Lines Distribution (Cleaned to Yes/No, then mapped to 0/1 if binary)
    # Note: Handled according to actual string values remaining in column
    # -------------------------------------------------------------------------
    logger.info("How many customers have Multiple Lines?")
    num_line_service = df['MultipleLines'].value_counts()
    print(num_line_service)
    logger.info("Percentage of customers there have Multiple Lines = 86.29%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='MultipleLines')
    plt.title('Customers with Multiple Lines')
    # Adjusted xticks dynamically based on categorical representation
    if df['MultipleLines'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Multiple Lines (0)', 'MultipleLines (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 8. Internet Service Types Distribution (Categorical - Pie Chart)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Internet Service types?")
    num_internet_service = df['InternetService'].value_counts()
    print(num_internet_service)
    logger.info("Percentage of Internet Service types : Fiber optic = 44.0% , DSL = 34.4% ")

    plt.figure(figsize=(6, 6))
    plt.pie(
        num_internet_service.values,
        labels=num_internet_service.index,
        autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title('Distribution of Internet Service Types')
    plt.show()

    # -------------------------------------------------------------------------
    # 9. Online Security Distribution (Cleaned and Mapped)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Online Security?")
    num_online_security = df['OnlineSecurity'].value_counts()
    print(num_online_security)
    logger.info("Percentage of customers there have Online Security = 28.6%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='OnlineSecurity')
    plt.title('Customers with Online Security')
    if df['OnlineSecurity'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Online Security (0)', 'Online Security (1)'])
    else:
        plt.xticks(rotation=0)
    plt.show()

    # -------------------------------------------------------------------------
    # 10. Online Backup Distribution (Cleaned and Mapped)
    # -------------------------------------------------------------------------
    logger.info("How many customers have Online Backup?")
    num_online_backup = df['OnlineBackup'].value_counts()
    print(num_online_backup)
    logger.info("Percentage of customers there have Online Backup = 34.4%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='OnlineBackup')
    plt.title('Customers with Online Backup')
    if df['OnlineBackup'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Online Backup (0)', 'Backup (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 11. Device Protection Distribution (Cleaned and Mapped)
    # -------------------------------------------------------------------------
    logger.info("How many customers have Device Protection?")
    num_device_protection = df['DeviceProtection'].value_counts()
    print(num_device_protection)
    logger.info("Percentage of customers there have Device Protection = 34.3%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='DeviceProtection')
    plt.title('Customers with Device Protection')
    if df['DeviceProtection'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not DeviceProtection (0)', 'DeviceProtection (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 12. Tech Support Distribution (Cleaned and Mapped)
    # -------------------------------------------------------------------------
    logger.info("How many customers have Tech Support?")
    num_tech_support = df['TechSupport'].value_counts()
    print(num_tech_support)
    logger.info("Percentage of customers there have Tech Support = 29.02%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='TechSupport')
    plt.title('Customers with Tech Support')
    if df['TechSupport'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Tech Support (0)', 'Tech Support (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 13. Streaming Services Distribution (Movies & TV)
    # -------------------------------------------------------------------------
    logger.info("How many customers use Streaming TV or Streaming Movies?")
    num_streaming_movies = df['StreamingMovies'].value_counts()
    print(num_streaming_movies)
    logger.info("Percentage of customers use Streaming Movies = 38.79%")

    num_streaming_TV = df['StreamingTV'].value_counts()
    print(num_streaming_TV)
    logger.info("Percentage of customers use Streaming TV = 38.4%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='StreamingMovies')
    plt.title('Customers with Streaming Movies')
    if df['StreamingMovies'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Streaming Movies (0)', 'Movies (1)'])
    plt.show()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='StreamingTV')
    plt.title('Customers using Streaming TV')
    if df['StreamingTV'].dtype in ['int64', 'float64']:
        plt.xticks([0, 1], ['Not Streaming TV (0)', 'Streaming TV (1)'])
    plt.show()

    # -------------------------------------------------------------------------
    # 14. Contract Types Distribution (Categorical Strings)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Contract types?")
    num_contract_type = df['Contract'].value_counts()
    print(num_contract_type)
    logger.info("Percentage of Contract types --->> \n Month-to-month: 55.0% \n 1 Year: 20.9% \n 2 Year: 24.1%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Contract')
    plt.title('Customers with Contract Type')
    plt.show()

    # -------------------------------------------------------------------------
    # 15. Payment Methods Distribution (Categorical Strings with Rotation)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Payment Methods?")
    num_payment_method = df['PaymentMethod'].value_counts()
    print(num_payment_method)
    logger.info("Percentage of Payment Methods --->> \n Electronic check: 33.6% \n Mailed check: 22.9% \n Bank transfer: 21.9% \n Credit card: 21.6%")

    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x='PaymentMethod')
    plt.title('Customers with Payment Method')
    plt.xticks(rotation=45)
    plt.tight_layout()  # Prevents cut-off labels due to rotation
    plt.show()

    # -------------------------------------------------------------------------
    # 16. Monthly Charges Distribution (Continuous Numerical)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Monthly Charges?")
    num_monthly_charges = df['MonthlyCharges'].describe()
    print(num_monthly_charges)
    logger.info("Insight: MonthlyCharges show that most customers pay around $65 per month, "
                "with a wide range between low-cost and premium plans. "
                "Customers paying higher charges (above $90) may have a higher churn risk.")

    plt.figure(figsize=(8, 4))
    plt.hist(df['MonthlyCharges'], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Monthly Charges")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # -------------------------------------------------------------------------
    # 17. Total Charges Distribution (Continuous Numerical)
    # -------------------------------------------------------------------------
    logger.info("How are Total Charges distributed among customers?")
    num_total_charges = df['TotalCharges'].describe()
    print(num_total_charges)
    logger.info("Insight: TotalCharges are widely spread, reflecting differences in customer tenure. "
                "Half of the customers have paid less than $1400, indicating a large group of newer subscribers.")

    plt.figure(figsize=(8, 4))
    plt.hist(df['TotalCharges'], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Total Charges")
    plt.xlabel("Total Charges")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    # -------------------------------------------------------------------------
    # 18. Paperless Billing Usage Distribution (Categorical - Pie Chart)
    # -------------------------------------------------------------------------
    logger.info("What is the distribution of Paperless Billing usage?")
    num_paperless_billing_usage = df['PaperlessBilling'].value_counts()
    print(num_paperless_billing_usage)
    logger.info("Percentage of Paperless Billing usage = 59.2%")

    plt.figure(figsize=(6, 6))
    plt.pie(
        num_paperless_billing_usage.values,
        labels=["No (0)", "Yes (1)"] if df['PaperlessBilling'].dtype in ['int64', 'float64'] else num_paperless_billing_usage.index,
        autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title("Distribution of Paperless Billing")
    plt.show()

    # -------------------------------------------------------------------------
    # 19. Target Variable Distribution: Churn vs Stayed (Mapped to 0/1)
    # -------------------------------------------------------------------------
    print('How many customers churned vs. stayed?')
    num_churned_vs_stayed = df['Churn'].value_counts()
    print(num_churned_vs_stayed)
    print("Percentage of customers churned: 26.5%")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution (0 = Stayed, 1 = Left)')
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks([0, 1], ['Stayed (0)', 'Churned (1)'])
    plt.show()

    logger.info("============ Basic EDA Completed ============")


if __name__ == "__main__":
    from data_pipeline import run_data_pipeline
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = run_data_pipeline(FILE_PATH)
    basic_eda(df)
