from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from other.articles import articles
import numpy as np
from utils import (
    array_to_str,
    get_patents_from_json,
    print_token_length,
    split_text_by_tokens,
)
import re

model = SentenceTransformer("all-mpnet-base-v2")
max_tokens = model.max_seq_length
tokenizer = model.tokenizer
embeddings = {}
  
def convert_article_to_embedding(content):
    """
    Input: Articles
    Converts each article into a shape (1,768) tensor
    """
    embedding = model.encode(content)
    return(embedding)

def compare_article_similarity(embeddings):
    """
    Input : Array of embeddings (n,768)
    Prints for each article how similar it is to every other article
    """
    for title,embedding in embeddings.items():
        embedding1 = embeddings[title].reshape(1, -1)
        print(embedding1.shape)
        print(f"Calculating document similarity against {title}")

        for title,embedding in embeddings.items():
            embedding2 = embeddings[title].reshape(1, -1)
            similarity = model.similarity_pairwise(embedding1, embedding2)
            print(f"    similarity score against {title} is {similarity}")

def compare_patent_article(patent_id,patent_embedding, article_dict):
    patent_embedding = patent_embedding.reshape(1, -1)

    for title,embedding in article_dict.items():
        embedding = embedding.reshape(1, -1)
        similarity = model.similarity_pairwise(patent_embedding,embedding)
        print(f"Similarity: {patent_id} against {title} | {similarity} ")

def get_average_embedding(patent_id,text,text_type):
    """
    Description:
    (1) Seperate text into chunks <= max token length
    (2) Get embedding for each chunk
    (3) Take average_embedding

    Input: patentid, text, text_type
    Output: average embedding (768,)
    """
    sentence_embeddings = []
    print_token_length(tokenizer,f"{patent_id} {text_type} : ",text)
    text_chunked = split_text_by_tokens(tokenizer,text,max_tokens)
    for chunk in text_chunked:
        embedding = model.encode(chunk)
        sentence_embeddings.append(embedding)
    print(f"{text_type} emb dim is {len(sentence_embeddings)}")
    sentence_embeddings= np.array(sentence_embeddings)
    average_embeddings = sentence_embeddings.mean(axis=0)
    return average_embeddings
    
def convert_patent_to_embedding(patent_id,patent):
    """
    Description: 
        (1) Extract abstract, claims and description from patent
        (2) Get Average of the abstract, claim and description embeddings
        
    Input: Patent object
    Output: 1 (1,768) embedding
    """

    abstract = patent.get("abstract")
    claims = array_to_str(patent.get("claims"))
    description = patent.get("description")

    abstract_embeddings = get_average_embedding(patent_id,abstract,"abstract")
    claim_embeddings = get_average_embedding(patent_id,claims,"claims")
    description_embeddings = get_average_embedding(patent_id,description,"description")

    embeddings = np.array([
        abstract_embeddings, claim_embeddings, description_embeddings
    ])
    average_embedding = embeddings.mean(axis=0)
    return average_embedding

def split_description_points(description):
    """
    Splits patent text into sections starting at [0001], [0002], etc.
    
    Returns:
        List of strings, where each item begins with its bracketed number.
    """
    pattern = r'(\[\d{4}\])'
    
    parts = re.split(pattern, description)
    sections = []

    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append(f"{marker} {content}".strip())
        i += 2

    return sections

def compute_embeddings(text_units):
    """
    Encode a list of text units and return their embeddings.

    Args:
        text_units: list[str]
        model: SentenceTransformer model

    Returns:
        list of embeddings
    """
    embeddings = []

    for text in text_units:
        if not text or not text.strip():
            continue

        embedding = model.encode(text)
        embeddings.append(embedding)

    return embeddings

def fine_patent_article_comparison(patent):
    print("func activate")

    description = split_description_points(patent.get("description"))
    claims = patent.get("claims")

    description_embeddings = compute_embeddings(description)
    print(len(description_embeddings))
    print("func works")

    claim_embeddings = compute_embeddings(claims)
    print(claim_embeddings)
    print("func breaks")

    chunk_counter = 0
    embedding_counter = 0

    for embedding in description_embeddings:
        embedding_counter += 1
        print("hello")

        for title, article_text in articles.items():
            chunks = split_text_by_tokens(tokenizer, article_text, max_tokens=80)
            print(f"{title} has {len(chunks)} chunks")

            chunk_embeddings = compute_embeddings(chunks)

            print("Whats up")
            for chunk_embedding in chunk_embeddings:
                chunk_counter += 1
                similarity = model.similarity(embedding, chunk_embedding)
                print(
                    f"Similarity between description {embedding_counter} "
                    f"and chunk {chunk_counter} of {title} | {similarity}"
                )

def main():
    patents = get_patents_from_json()
    patent = patents[0]
    patent_id = patents[0]["patent_id"]

    patent_embedding = convert_patent_to_embedding(patent_id,patent)
    article_dict = {}

    for title,content in articles.items():
        article_embedding = convert_article_to_embedding(content)
        article_dict[str(title)] = article_embedding

    compare_patent_article(patent_id, patent_embedding, article_dict)
    content = patent["description"]
    fine_patent_article_comparison(patent)        

main()





# convert_articles_to_embeddings(articles)
# compare_article_similarity(embeddings)

# calc window size using: model.max_seq_length
# similarity = model.similarity(embeddings,embeddings)


# 384 is the max token length of sentence transformer : all-mpnet-base-v2