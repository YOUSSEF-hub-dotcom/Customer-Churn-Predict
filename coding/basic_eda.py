import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("EDA_1")

def basic_eda(df):
    logger.info("============ Exploratory Data Analysis (Basic) & Visualization ============")

    logger.info("What is the distribution of gender in the dataset?")
    dist_gender = df['gender'].value_counts()
    print(dist_gender)
    logger.info("Distribution of Female: 49.6% , Male: 50.4%")

    sns.countplot(data=df, x='gender')
    plt.title('Distribution of Gender')
    plt.show()


    logger.info("How many customers are Senior Citizens?")
    num_senior_citizens = df['SeniorCitizen'].value_counts()
    print(num_senior_citizens)
    logger.info("Percentage of customers Senior Citizens (olderly people) = 16.2%")

    sns.countplot(data=df, x='SeniorCitizen')
    plt.title('Distribution of Senior Citizens')
    plt.xticks([0, 1], ['Not Senior Citizen (0)', 'Senior Citizen (1)'])
    plt.show()


    logger.info("How many customers have partners?")
    num_partners = df['Partner'].value_counts()
    print(num_partners)
    logger.info("Percentage of customers Partners = 48.3%")

    sns.countplot(data=df, x='Partner')
    plt.title('Customers with Partner')
    plt.xticks([0, 1], ['Not Partners (0)', 'Partners (1)'])
    plt.show()


    logger.info("How many customers have dependents?")
    num_dependents = df['Dependents'].value_counts()
    print(num_dependents)
    logger.info("Percentage of customers Dependents = 29.9%")

    sns.countplot(data=df, x='Dependents')
    plt.title('Customers with Dependents')
    plt.xticks([0, 1], ['Not Dependents (0)', 'Dependents (1)'])
    plt.show()


    logger.info("What is the distribution of tenure?")
    num_tenure = df['tenure'].describe()
    print(num_tenure)
    logger.info("The tenure of customers ranges from 0 to 72 months, "
          "with a mean of 32 months. Most customers have a tenure around 2-3 years. "
          "New customers (tenure ≤ 12 months) are more likely to churn, "
          "while long-term customers (tenure ≥ 49 months) tend to be loyal.")

    sns.histplot(data=df, x='tenure', bins=30, kde=True)
    plt.title('Distribution of Tenure')
    plt.show()


    logger.info("How many customers have Phone Service?")
    num_phone_service = df['PhoneService'].value_counts()
    print(num_phone_service)
    logger.info("Percentage of customers there have Phone Services = 90.3%")

    sns.countplot(data=df, x='PhoneService')
    plt.title('Customers with Phone Service')
    plt.xticks([0, 1], ['Not PhoneService (0)', 'PhoneService (1)'])
    plt.show()


    logger.info("How many customers have Multiple Lines?")
    num_line_service = df['MultipleLines'].value_counts()
    print(num_line_service)
    logger.info("Percentage of customers there have Multiple Lines = 86.29%")

    sns.countplot(data=df, x='MultipleLines')
    plt.title('Customers with Multiple Lines')
    plt.xticks([0, 1], ['Not Multiple Lines (0)', 'MultipleLines (1)'])
    plt.show()


    logger.info("What is the distribution of Internet Service types?")
    num_internet_service = df['InternetService'].value_counts()
    print(num_internet_service)
    logger.info("Percentage of Internet Service types : Fiber optic = 44.0% , DSL = 34.4% ")

    plt.pie(
        num_internet_service.values,
        labels=num_internet_service.index,
        autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black'}
    )
    plt.title('Distribution of Internet Service Types')
    plt.show()


    logger.info("What is the distribution of Online Security?")
    num_online_security = df['OnlineSecurity'].value_counts()
    print(num_online_security)
    logger.info("Percentage of customers there have Online Security = 28.6%")

    sns.countplot(data=df, x='OnlineSecurity')
    plt.title('Customers with Online Security')
    plt.xticks([0, 1], ['Not Online Security (0)', 'Online Security (1)'])
    plt.show()


    logger.info("How many customers have Online Backup?")
    num_online_backup = df['OnlineBackup'].value_counts()
    print(num_online_backup)
    logger.info("Percentage of customers there have Online Backup = 34.4%")

    sns.countplot(data=df, x='OnlineBackup')
    plt.title('Customers with Online Backup')
    plt.xticks([0, 1], ['Not Online Backup (0)', 'Backup (1)'])
    plt.show()


    logger.info("How many customers have Device Protection?")
    num_device_protection = df['DeviceProtection'].value_counts()
    print(num_device_protection)
    logger.info("Percentage of customers there have Device Protection = 34.3%")

    sns.countplot(data=df, x='DeviceProtection')
    plt.title('Customers with Device Protection')
    plt.xticks([0, 1], ['Not DeviceProtection (0)', 'DeviceProtection (1)'])
    plt.show()


    logger.info("How many customers have Tech Support?")
    num_tech_support = df['TechSupport'].value_counts()
    print(num_tech_support)
    logger.info("Percentage of customers there have Tech Support = 29.02%")

    sns.countplot(data=df, x='TechSupport')
    plt.title('Customers with Tech Support')
    plt.xticks([0, 1], ['Not Tech Support (0)', 'Tech Support (1)'])
    plt.show()


    logger.info("How many customers use Streaming TV or Streaming Movies?")
    num_streaming_movies = df['StreamingMovies'].value_counts()
    print(num_streaming_movies)
    logger.info("Percentage of customers use Streaming Movies = 38.79%")

    num_streaming_TV = df['StreamingTV'].value_counts()
    print(num_streaming_TV)
    logger.info("Percentage of customers use Streaming TV = 38.4%")

    sns.countplot(data=df, x='StreamingMovies')
    plt.title('Customers with Streaming Movies')
    plt.xticks([0, 1], ['Not Streaming Movies (0)', 'Movies (1)'])
    plt.show()

    sns.countplot(data=df, x='StreamingTV')
    plt.title('Customers using Streaming TV')
    plt.xticks([0, 1], ['Not Streaming TV (0)', 'Streaming TV (1)'])
    plt.show()


    logger.info("What is the distribution of Contract types?")
    num_contract_type = df['Contract'].value_counts()
    print(num_contract_type)
    logger.info("Percentage of Contract types --->> \n Month-to-month: 55.0% \n 1 Year: 20.9% \n 2 Year: 24.1%")

    sns.countplot(data=df, x='Contract')
    plt.title('Customers with Contract Type')
    plt.show()


    logger.info("What is the distribution of Payment Methods?")
    num_payment_method = df['PaymentMethod'].value_counts()
    print(num_payment_method)
    logger.info("Percentage of Payment Methods --->> \n Electronic check: 33.6% \n Mailed check: 22.9% \n Bank transfer: 21.9% \n Credit card: 21.6%")

    sns.countplot(data=df, x='PaymentMethod')
    plt.title('Customers with Payment Method')
    plt.xticks(rotation=45)
    plt.show()


    logger.info("What is the distribution of Monthly Charges?")
    num_monthly_charges = df['MonthlyCharges'].describe()
    print(num_monthly_charges)
    logger.info("Insight: MonthlyCharges show that most customers pay around $65 per month, "
          "with a wide range between low-cost and premium plans. "
          "Customers paying higher charges (above $90) may have a higher churn risk.")

    plt.hist(df['MonthlyCharges'], bins=30)
    plt.title("Distribution of Monthly Charges")
    plt.xlabel("Monthly Charges")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()


    logger.info("How are Total Charges distributed among customers?")
    num_total_charges = df['TotalCharges'].describe()
    print(num_total_charges)
    logger.info("Insight: TotalCharges are widely spread, reflecting differences in customer tenure. "
          "Half of the customers have paid less than $1400, indicating a large group of newer subscribers.")

    plt.hist(df['TotalCharges'], bins=30)
    plt.title("Distribution of Total Charges")
    plt.xlabel("Total Charges")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()


    logger.info("What is the distribution of Paperless Billing usage?")
    num_paperless_billing_usage = df['PaperlessBilling'].value_counts()
    print(num_paperless_billing_usage)
    logger.info("Percentage of Paperless Billing usage = 59.2%")

    plt.pie(
        num_paperless_billing_usage.values,
        labels=["No (0)", "Yes (1)"],
        autopct='%1.1f%%'
    )
    plt.title("Distribution of Paperless Billing")
    plt.show()


    print('How many customers churned vs. stayed?')
    num_churned_vs_stayed = df['Churn'].value_counts()
    print(num_churned_vs_stayed)
    print("Percentage of customers churned: 26.5%")

    sns.countplot(data=df, x='Churn')
    plt.title('Churn Distribution (0 = Stayed, 1 = Left)')
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.show()

    logger.info("============ Basic EDA Completed ============")


if __name__ == "__main__":
    from data_pipeline import run_data_pipeline
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = run_data_pipeline(FILE_PATH)
    basic_eda(df)
