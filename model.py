import joblib
import os
import pandas as pd
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

global account_size_stats
account_size_stats = {
    "median": {
        "little": 125.0, 
        "moderate": 130.5, 
        "no data": 160.0, 
        "quite rich": 118.0, 
        "rich": 120.0
    }, 
    "std": {
        "little": 75.0, 
        "moderate": 85.0, 
        "no data": 90.0, 
        "quite rich": 63.0, 
        "rich": 53.0
    }
}

# Model handling
def model_fn(model_dir):
    """Load the model for inference."""
    model = joblib.load(os.path.join(model_dir, "voting_classifier.pkl"), mmap_mode=None)
    return model

def input_fn(request_body, request_content_type):
    # Deserialize the input data
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        # Apply the transformation
        logging.info(f"Raw input_data type in input_fn: {type(input_data)}")
        logging.info(f"Raw input_data in input_fn: {input_data}")
        logging.info(f"Raw input_data shape: {len(input_data)}")
        transformed_data = transform_input_data_to_call_endpoint(input_data)
        logging.info(f"Raw input_data type in input_fn: {type(transformed_data)}")
        logging.info(f"Raw input_data in input_fn: {transformed_data}")
        logging.info(f"Raw input_data type in input_fn: {transformed_data.shape}")
        return transformed_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make predictions."""
    logging.info("Starting prediction process.")
    logging.info(f"Raw input_data type: {type(input_data)}")
    logging.info(f"Raw input_data shape: {len(input_data)}")
    prediction = model.predict(input_data)

    if len(prediction)==1:
        return [prediction[0]]

    return prediction


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

def get_monthly_payment_stats(account_size, monthly_payment):
    mean = account_size_stats["median"][account_size]
    std = account_size_stats["std"][account_size]
    if monthly_payment > mean + std:
        return "high"
    else:
        return "normal"

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