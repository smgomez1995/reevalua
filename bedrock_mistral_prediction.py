import json
import time

import boto3
from tqdm.notebook import tqdm

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "mistral.mistral-small-2402-v1:0"


def call_model(model_id, prompt):
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=f'{{"prompt":"<s>[INST] {prompt} [/INST]", "max_tokens":12, "temperature":0, "top_p":1, "top_k":10}}',
    )

    return response["body"].read().decode("utf-8")


def return_prompt(job, housing, category, account, high_payment, months):
    return f"You are a credit analyst, deciding if a given loan is good or bad. 'good', the loan should be paid without issues. 'bad', it is probable that it is paid in delay or not paid at all. Consider the following variables to decide: Type of job: '{job}'. Options ['unskilled and non-resident', 'unskilled and resident', 'skilled', 'highly skilled']. Housing Situation: '{housing}'. Options: ['free', 'own', 'rent']. Loan Category: '{category}'. Options: ['self development', 'maintenance', 'recreational']. Account Size: '{account}'. Options: ['no data', 'little', 'moderate', 'rich', 'quite rich']. High Monthly Payment: '{high_payment}'. Options: ['normal', 'high']. Duration (in months): {months}. Just respond in json format, with the 'loan': 'bad' or 'good'."


def classify_using_llm(df_converted):
    responses = []
    for i, row in tqdm(df_converted.iterrows(), total=len(df_converted)):
        job = row["Job"]
        housing = row["Housing"]
        category = row["Loan Category"]
        account = row["Account Size"]
        high_payment = row["High Monthly_Payment"]
        months = row["Duration"]
        kwargs = {
            "job": job,
            "housing": housing,
            "category": category,
            "account": account,
            "high_payment": high_payment,
            "months": months,
        }
        prompt = return_prompt(**kwargs)

        retries = 0
        while retries < 5:
            try:
                response = call_model(model_id, prompt)
                responses.append(response)
                break
            except client.exceptions.ThrottlingException:
                wait_time = 2**retries
                print(f"{i} ThrottlingException: Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
        else:
            print("Failed to get response after multiple retries.")
            responses.append(None)

        time.sleep(5)

    clasifications = []
    for response in responses:
        try:
            clasification = json.loads(json.loads(response)["outputs"][0]["text"])[
                "loan"
            ]
        except:
            string_with_info = (
                json.loads(response)["outputs"][0]["text"]
                .replace("{\n", "")
                .replace("risky", 'risky"')
                .strip()
            )
            clasification = string_with_info.split(":")[1].strip().replace('"', "")
        clasifications.append(clasification)

    df_converted["Prediction Mistral Small"] = clasifications
    # there is a little bit hallucination in the model´s answers, so we need to fix it
    df_converted["Prediction Mistral Small"] = df_converted[
        "Prediction Mistral Small"
    ].replace("potentially risky", "bad")

    df_converted.to_csv("credit_risk_reto_processed.csv", index=False)

    return df_converted
