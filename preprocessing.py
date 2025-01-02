import pandas as pd
import json
import numpy as np

JOB_DESCRIPTION = {
    0: "unskilled and non-resident",
    1: "unskilled and resident",
    2: "skilled",
    3: "highly skilled",
}

ACCOUNT_CATEGORY_CONVERSION = {
    "no data": 0,
    "little": 1,
    "moderate": 2,
    "rich": 3,
    "quite rich": 4,
}

TRANSFORMED_COLUMNS = [
    "Account Size_little",
    "Account Size_moderate",
    "Account Size_no data",
    "Account Size_quite rich",
    "Account Size_rich",  
    "Age Group_0_20",
    "Age Group_21_30",
    "Age Group_31_40",
    "Age Group_41_50",
    "Age Group_51_60",
    "Age Group_61_70",
    "Age Group_71_80",  
    "Credit amount",  
    "Duration",  
    "High Monthly_Payment_high",
    "High Monthly_Payment_normal",  
    "Housing_free",
    "Housing_own",
    "Housing_rent",  
    "Job_highly skilled",
    "Job_skilled",
    "Job_unskilled and non-resident",
    "Job_unskilled and resident",  
    "Purpose_business",
    "Purpose_car",
    "Purpose_domestic appliances",
    "Purpose_education",
    "Purpose_furniture/equipment",
    "Purpose_radio/TV",
    "Purpose_repairs",
    "Purpose_vacation/others",
]

ORIGINAL_COLUMNS = [
    'Age', 
    'Sex', 
    'Job', 
    'Housing', 
    'Saving accounts', 
    'Checking account',
    'Credit amount', 
    'Duration', 
    'Purpose'
]

def get_max_account(columns):
    # we convert the account category to number representation
    savings_account = columns[0]
    checking_account = columns[1]

    max_account = max(
        ACCOUNT_CATEGORY_CONVERSION.get(savings_account, 0),
        ACCOUNT_CATEGORY_CONVERSION.get(checking_account, 0),
    )

    return [
        key
        for key, value in ACCOUNT_CATEGORY_CONVERSION.items()
        if value == max_account
    ][0]


def categorize_loan(purpose):
    # we add some "knowledge" of the purpose type
    if purpose in ["radio/TV", "vacation/others"]:
        return "recreational"
    elif purpose in ["education", "business", "car"]:
        return "development"
    elif purpose in ["furniture/equipment", "domestic appliances", "repairs"]:
        return "maintenance"


def remove_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series[(series >= lower_bound) & (series <= upper_bound)]


def get_account_size_stats():
    df = pd.read_csv("credit_risk_reto.csv")
    df.fillna("no data", inplace=True)
    df["Monthly_Payment"] = (df["Credit amount"] / df["Duration"]).round(0)
    df["Account Size"] = df[["Saving accounts", "Checking account"]].apply(
        get_max_account, axis=1
    )
    df_no_outliers = (
        df.groupby("Account Size")["Monthly_Payment"]
        .apply(remove_outliers)
        .reset_index(level=0)
    )
    account_size_stats = (
        df_no_outliers.groupby("Account Size")["Monthly_Payment"]
        .agg(["median", "std"])
        .round({"std": 0})
        .to_dict()
    )
    return account_size_stats

def check_or_create_account_size_stats():
    try:
        with open("account_size_stats.json") as json_file:
            account_size_stats = json.load(json_file)
    except FileNotFoundError:
        account_size_stats = get_account_size_stats()
        with open("account_size_stats.json", "w") as json_file:
            json.dump(account_size_stats, json_file)
    return account_size_stats

global account_size_stats
account_size_stats = check_or_create_account_size_stats()


def get_monthly_payment_stats(account_size, monthly_payment):
    mean = account_size_stats["median"][account_size]
    std = account_size_stats["std"][account_size]
    if monthly_payment > mean + std:
        return "high"
    else:
        return "normal"


def pre_processing_data(file_path):
    global account_size_stats

    df = pd.read_csv(file_path)

    # after some analysis, the only columns with nan is the "Saving accounts"
    # and "Checking account" columns. We assumed that this is always true
    df.fillna("no data", inplace=True)

    # convert job from int to string
    df["Job"] = df["Job"].replace(JOB_DESCRIPTION)
    df["Monthly_Payment"] = (df["Credit amount"] / df["Duration"]).round(0)
    df["Loan Category"] = df["Purpose"].apply(lambda x: categorize_loan(x))

    # Define age bins and labels
    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_labels = [
        "0_20",
        "21_30",
        "31_40",
        "41_50",
        "51_60",
        "61_70",
        "71_80",
        "81_90",
        "91_100",
    ]

    # Create a new column 'Age Group' with the age clusters
    df["Age Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    string_columns = df.select_dtypes(include=["object", "category"]).columns
    for col in string_columns:
        df[col] = df[col].astype("category")

    # Get the bigger account size
    df["Account Size"] = df[["Saving accounts", "Checking account"]].apply(
        get_max_account, axis=1
    )

    df["High Monthly_Payment"] = (
        df[["Account Size", "Monthly_Payment"]]
        .apply(lambda x: get_monthly_payment_stats(*x), axis=1)
        .astype("category")
    )
    df.drop(columns=["Saving accounts", "Checking account", "Age"], inplace=True)

    return df


def transform_input_data_to_call_endpoint(df):
    global account_size_stats
    if isinstance(df, list):
        df = pd.DataFrame(df)
    elif isinstance(df, np.ndarray):
        df = pd.DataFrame(df, columns=ORIGINAL_COLUMNS)
    if df.empty:
        return pd.DataFrame()

    age_labels = [
        "0_20",
        "21_30",
        "31_40",
        "41_50",
        "51_60",
        "61_70",
        "71_80"
    ]

    input_data_transformed = []

    for i, row in df.iterrows():
        dict_transform_to_df = {}
        # age
        age_group_for_row = [
            "Age Group_" + range_bin
            for range_bin in age_labels
            if int(range_bin.split("_")[0])
            <= row["Age"]
            <= int(range_bin.split("_")[1])
        ][0]
        dict_transform_to_df[age_group_for_row] = 1

        # job
        dict_transform_to_df["Job_" + JOB_DESCRIPTION[row["Job"]]] = 1

        # housing
        dict_transform_to_df["Housing_" + row["Housing"]] = 1

        # saving accounts and checking account
        accounts = get_max_account([row["Saving accounts"], row["Checking account"]])
        dict_transform_to_df["Account Size_" + accounts] = 1

        # credit amount
        dict_transform_to_df["Credit amount"] = row["Credit amount"]

        # duration
        dict_transform_to_df["Duration"] = row["Duration"]

        # new variable high monthly payment
        montly_payment = row["Credit amount"] / row["Duration"]
        dict_transform_to_df[
            "High Monthly_Payment_"
            + get_monthly_payment_stats(accounts, montly_payment)
        ] = 1

        # purpose
        dict_transform_to_df["Purpose_" + row["Purpose"]] = 1

        input_data_transformed.append(dict_transform_to_df)

    input_data_transformed = pd.DataFrame(input_data_transformed).fillna(0.0)
    assert len(
        set(input_data_transformed.columns).intersection(set(TRANSFORMED_COLUMNS))
    ) == len(input_data_transformed.columns), "New Values are trying to be transformed"

    missing_cols = set(TRANSFORMED_COLUMNS) - set(input_data_transformed.columns)
    for col in missing_cols:
        input_data_transformed[col] = 0

    input_data_transformed = input_data_transformed[TRANSFORMED_COLUMNS]

    return input_data_transformed