# Ticket Segmentation

Email support messages are not segmented automatically in zendesk, however if you have a dataset of you manually segmenting you can train an ML model to segment the data for you!

I built on two models. 

## OpenAI

I fine-tune the OpenAI gpt-3.5 model for ticket segmentation. 

### Usage

1. Upload your data in the form of json in the data.jsonl file.
2. Run `fine-tune.py` to create and fine-tune a model with the data you provided.
3. Get the ID of the model.
4. Run `main.py` with the id of the model.

## LLaMA

An open-source model that has amazing chat ability. I fine-tune the 7B parameter model with LoRA. 

1. Upload your data to `example.csv`.
2. Fine-tune the model with LoRA. 
3. Run inference on the model with the current email.