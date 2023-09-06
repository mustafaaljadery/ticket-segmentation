import os
import openai


def generate_prompt(data):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
    ### Context:
    {data["instruction"]}
    ### Input:
    {data["input"]}
    ### Response:
    """


def main(message):
    openai.apy_key = os.getenv("OPENAI_API_KEY")

    data = {
        "instruction": "Detect the label of this support ticket. These are the following options: Buy or software issue, Creating a post, Can't access my account, Billing, Editor, Upgrading account, Migrating from a different platform, Feature request, Something else, Delete my account.",
        "input": message[0],
    }

    completion = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo:my-org:custom_suffix:id",
        messages=[
            {"role": "system",
                "content": "You're the best customer support agent in the world!"},
            {"role": "user", "content": generate_prompt(data)},
        ]
    )
    return completion.choices[0].message
