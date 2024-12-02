from dotenv import load_dotenv
import os

from transformers import (
    pipeline,
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch


load_dotenv()
token = os.getenv('TOKEN')

quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", token=token
)
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=quantization,
    token=token,
    device_map="auto",
)
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)


def generate(prompt):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        return_full_text=False,
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful chatbot. Answer to the questions in a helpful way",
        },
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = pipe(prompt)
    return response[0]["generated_text"]
