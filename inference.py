from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

def prepare_model(trained_model_path= './weights/checkpoint-39000'):
    model_tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
    trained_model = AutoModelForSeq2SeqLM.from_pretrained(trained_model_path)
    return trained_model, model_tokenizer

# process function:
def process_function(text_inputs, model_tokenizer):
    prefix = "translate English to Vietnamese:"
    processed_text = [prefix + str(x) for x in text_inputs]
    return [
        model_tokenizer(x,
                        max_length=128,
                        truncation=True,
                        return_tensors="pt").input_ids for x in processed_text
    ]

def translated_fn(model, input_texts, tokenizer):
    process_texts = process_function(input_texts, tokenizer)
    model_outputs = [
        model.generate(model_input,
                       max_new_tokens=40,
                       do_sample=True,
                       top_k=30,
                       top_p=0.95) for model_input in process_texts
    ]
    return [
        tokenizer.decode(output[0], skip_special_tokens=True)
        for output in model_outputs
    ]