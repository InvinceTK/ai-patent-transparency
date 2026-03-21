import json
def print_token_length(tokenizer, title,content):
    """
    Prints token length of an individual article
    """
    print(type(content))
    tokens = tokenizer(content)
    token_length = len(tokens["input_ids"])
    print(f"The article titled : {title} has {token_length} tokens")

# Reading some text
def array_to_str(arr):
    """
    Description : Turns an array of claims into one string of patents
    Input: Patent
    Output: string(Patent)
    """

    text = ""

    for item in arr:
        text += item + "\n"
    
    return text

def get_patents_from_json():
    """
    Description: Reads patent data from sample.json

    Output: 
    patent[]
    
    Patent object:
        patent_id,
        bilio
        abstract
        description
        claims

    """

    with open ("sample.json","r") as f:
        data = json.load(f)

    return data

# split text
def split_text_by_tokens(tokenizer, text, max_tokens):

    """
    Description : Splits text according to max_token_length
    Input : abstract | claims | description
    Output: chunk[]
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks
