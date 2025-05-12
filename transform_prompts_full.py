import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to rewrite a single query using Qwen
def rewrite_query_with_qwen(query):
    # Construct prompt with examples and the user's query
    user_prompt = (
        "For each input query, rewrite it into a question starting with “When is” or “When are,” followed by a detailed object description exactly as mentioned in the query, including any colors, materials, or spatial relations. If the query refers to an object in relation to another (e.g. “the blue jacket next to the TV”), preserve all objects and their relationships. Do not omit any descriptive detail."
        "For example:\n"
        "- 'What color is the chair in the kitchen?' -> 'When is the chair in the kitchen shown?'\n"
        "- 'Where is the trash can located?' -> 'When is the trash can shown?'\n"
        "- 'How many pillows on the left side of the bed that is located near the sofa chair?' -> "
        "'When is the bed that is located near the sofa chair shown?'\n"
        "- 'What color is the blue jacket next to the TV?' -> 'When is the blue jacket next to the TV shown?'\n"
        f"Now rewrite this question: {query}"
    )
    
    # Chat format expected by Qwen2.5
    messages = [
        {"role": "system", "content": "You are a helpful assistant that rewrites natural language queries into temporal grounding prompts for a moment retrieval model. "},
        {"role": "user", "content": user_prompt}
    ]

    # Create prompt for the model
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize and move input to the model device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract only the new output part (after the input prompt)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode output to text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Post-processing: remove quotation marks if model returns them
    response = response.strip('"').strip("'")

    return response

# File paths
input_file = 'eval.jsonl'
output_file = 'eval_transformed_full.jsonl'

# Open and process the file
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in tqdm(infile, desc="Processing queries"):
        # Load a single JSON line
        item = json.loads(line)
        
        # Extract the query
        original_query = item['query']
        
        # Rewrite query using Qwen
        try:
            rewritten_query = rewrite_query_with_qwen(original_query)
        except Exception as e:
            print(f"Error processing query '{original_query}': {e}")
            rewritten_query = original_query  # Fallback in case of failure

        # Replace the original query with the rewritten query
        item['query'] = rewritten_query

        # Save the modified line to the output file
        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"✅ All queries processed and saved to {output_file}")