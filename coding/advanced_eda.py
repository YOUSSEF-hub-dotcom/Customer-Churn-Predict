import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

logger = logging.getLogger("EDA_2")

def advanced_eda(df):
    logger.info("============ Advanced Exploratory Data Analysis & Churn Insights ============")

    logger.info("What is the impact of “Contract Type” on Customer Churn?")
    contract_churn = df.groupby(['Contract', 'Churn']).size().unstack()
    contract_churn_percent = contract_churn.div(contract_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Contract Type:")
    print(contract_churn)
    logger.info("Churn percentage by Contract Type:")
    print(contract_churn_percent)

    contract_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Contract Type on Customer Churn")
    plt.xlabel("Contract Type")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Does having “Online Security” reduce the likelihood of churn?")
    security_churn = df.groupby(['OnlineSecurity', 'Churn']).size().unstack()
    security_churn_percent = security_churn.div(security_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Online Security:")
    print(security_churn)
    logger.info("Percentage of Online Security:")
    print(security_churn_percent)

    security_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Online Security on Customer Churn")
    plt.xlabel("Online Security (0 = No, 1 = Yes)")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("How does “Tech Support” availability affect churn rates?")
    tech_churn = df.groupby(['TechSupport', 'Churn']).size().unstack()
    tech_churn_percent = tech_churn.div(tech_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Tech Support:")
    print(tech_churn)
    logger.info("Percentage of Tech Support:")
    print(tech_churn_percent)

    tech_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Tech Support on Customer Churn")
    plt.xlabel("Tech Support (0 = No, 1 = Yes)")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Are customers with 'Fiber Optic' internet more likely to churn than DSL users?")
    internet_churn = df.groupby(['InternetService', 'Churn']).size().unstack()
    internet_churn_percent = internet_churn.div(internet_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Internet Service:")
    print(internet_churn)
    logger.info("Percentage by Internet Service:")
    print(internet_churn_percent)

    internet_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Internet Service Type on Customer Churn")
    plt.xlabel("Internet Service Type")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Do customers with higher “Monthly Charges” have higher churn?")
    bins = [0, 35, 70, 120]
    labels = ['Low', 'Medium', 'High']
    df['MonthlyChargesCategory'] = pd.cut(df['MonthlyCharges'], bins=bins, labels=labels)

    monthly_churn = df.groupby(['MonthlyChargesCategory', 'Churn']).size().unstack()
    monthly_churn_percent = monthly_churn.div(monthly_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Monthly Charges Category:")
    print(monthly_churn)
    logger.info("Percentage by Monthly Charges Category:")
    print(monthly_churn_percent)

    monthly_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Monthly Charges on Customer Churn")
    plt.xlabel("Monthly Charges Category")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Impact of Total Charges on Customer Churn")
    bins = [0, 500, 1500, 3000, 9000]
    labels = ['Very Low', 'Low', 'Medium', 'High']
    df['TotalChargesCategory'] = pd.cut(df['TotalCharges'], bins=bins, labels=labels)

    total_churn = df.groupby(['TotalChargesCategory', 'Churn']).size().unstack()
    total_churn_percent = total_churn.div(total_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Total Charges Category:")
    print(total_churn)
    logger.info("Percentage by Total Charges Category:")
    print(total_churn_percent.round(2))

    total_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Total Charges on Customer Churn")
    plt.xlabel("Total Charges Category")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("How does 'Tenure' influence churn probability?")
    bins = [0, 12, 24, 48, 72]
    labels = ['0-12 months', '13-24 months', '25-48 months', '49-72 months']
    df['TenureCategory'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    tenure_churn = df.groupby(['TenureCategory', 'Churn']).size().unstack()
    tenure_churn_percent = tenure_churn.div(tenure_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Tenure Category:")
    print(tenure_churn)
    logger.info("Percentage by Tenure Category:")
    print(tenure_churn_percent)

    tenure_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Tenure on Customer Churn")
    plt.xlabel("Tenure Category")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Does 'Payment Method' influence churn?")
    payment_churn = df.groupby(['PaymentMethod', 'Churn']).size().unstack()
    payment_churn_percent = payment_churn.div(payment_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Payment Method:")
    print(payment_churn)
    logger.info("Percentage by Payment Method:")
    print(payment_churn_percent)

    payment_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Payment Method on Customer Churn")
    plt.xlabel("Payment Method")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Does the presence of Paperless Billing affect customer churn?")
    paperless = df.groupby(['PaperlessBilling', 'Churn']).size().unstack()
    paperless_percent = paperless.div(paperless.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Paperless Billing:")
    print(paperless)
    logger.info("Percentage by Paperless Billing:")
    print(paperless_percent)

    paperless_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Paperless Billing on Customer Churn")
    plt.xlabel("Paperless Billing")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Do customers with multiple services churn less?")
    df['NumServices'] = (
        (df['PhoneService'] == 1).astype(int) +
        (df['InternetService'] != 'No').astype(int) +
        (df['StreamingTV'] == 1).astype(int) +
        (df['StreamingMovies'] == 1).astype(int)
    )

    services_churn = df.groupby(['NumServices', 'Churn']).size().unstack(fill_value=0)
    services_churn_percent = services_churn.div(services_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Number of Services:")
    print(services_churn)
    logger.info("Percentage by Number of Services:")
    print(services_churn_percent)

    services_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Number of Services on Customer Churn")
    plt.xlabel("Number of Services")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Do customers with 'Device Protection' have lower churn rates?")
    device_protection_churn = df.groupby(['DeviceProtection', 'Churn']).size().unstack()
    device_protection_churn_percent = device_protection_churn.div(device_protection_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by Device Protection:")
    print(device_protection_churn)
    logger.info("Percentage by Device Protection:")
    print(device_protection_churn_percent)

    device_protection_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of Device Protection on Customer Churn")
    plt.xlabel("Device Protection")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("How does 'TechSupport + OnlineSecurity' combination affect churn?")
    df['TechSupport_OnlineSecurity'] = df['TechSupport'].astype(str) + "_" + df['OnlineSecurity'].astype(str)

    combo_churn = df.groupby(['TechSupport_OnlineSecurity', 'Churn']).size().unstack()
    combo_churn_percent = combo_churn.div(combo_churn.sum(axis=1), axis=0) * 100

    logger.info("Churn count by TechSupport + OnlineSecurity combination:")
    print(combo_churn)
    logger.info("Percentage by combination:")
    print(combo_churn_percent)

    combo_churn_percent.plot(kind='bar', stacked=True, color=['#5DADE2', '#E74C3C'])
    plt.title("Impact of TechSupport + OnlineSecurity on Customer Churn")
    plt.xlabel("TechSupport + OnlineSecurity (0=No, 1=Yes)")
    plt.ylabel("Percentage of Customers")
    plt.legend(["Stayed (0)", "Churned (1)"], loc='upper right')
    plt.grid(axis='y')
    plt.show()


    logger.info("Does InternetService type combined with MonthlyCharges affect churn?")
    df['ChargesSegment'] = pd.cut(df['MonthlyCharges'], bins=[0, 40, 70, 150], labels=['Low', 'Medium', 'High'])
    df['Internet_Charges'] = df['InternetService'].astype(str) + "_" + df['ChargesSegment'].astype(str)

    internet_charges_churn = df.groupby('Internet_Charges')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100

    logger.info("Percentage of Churn by InternetService + MonthlyCharges Segment:")
    print(internet_charges_churn)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=internet_charges_churn.reset_index(), x='Internet_Charges', y=1, palette='Reds')
    plt.title("Churn % by Internet Service + Monthly Charges Segment")
    plt.xlabel("Internet Service + Charges Category")
    plt.ylabel("Churn Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    logger.info("Are customers with short tenure but high monthly charges the most likely to churn?")
    df['TenureCategory'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
    df['MonthlyChargesCategory'] = pd.cut(df['MonthlyCharges'], bins=[0, 35, 70, 120], labels=["Low", "Medium", "High"])

    df = df.dropna(subset=['TenureCategory', 'MonthlyChargesCategory'])

    df['Tenure_Monthly'] = df['TenureCategory'].astype(str) + "_" + df['MonthlyChargesCategory'].astype(str)
    tenure_monthly_churn = df.groupby('Tenure_Monthly')['Churn'].value_counts(normalize=True).unstack().fillna(0) * 100

    logger.info("Percentage of Churn by Tenure + Monthly Charges combination:")
    print(tenure_monthly_churn)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=tenure_monthly_churn.reset_index(), x='Tenure_Monthly', y=1, palette='Reds')
    plt.title("Churn % by Tenure + Monthly Charges Segment")
    plt.xlabel("Tenure + Monthly Charges Category")
    plt.ylabel("Churn Percentage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


    logger.info("Are senior citizens more likely to churn compared to younger customers?")
    churn_rates = df.groupby('SeniorCitizen')['Churn'].value_counts(normalize=True).unstack() * 100
    print(churn_rates)

    churn_rates.plot(kind='bar', color=['#5DADE2', '#E74C3C'], figsize=(8, 5))
    plt.title("Churn Rate: Senior Citizens vs Younger Customers")
    plt.xlabel("SeniorCitizen (0 = Not Senior, 1 = Senior)")
    plt.ylabel("Percentage (%)")
    plt.legend(["Stayed (0)", "Churned (1)"])
    plt.show()

    logger.info("============ Advanced EDA Completed ============")


if __name__ == "__main__":
    from data_pipeline import run_data_pipeline
    FILE_PATH = r"C:\Users\Hedaya_city\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = run_data_pipeline(FILE_PATH)
    advanced_eda(df)
