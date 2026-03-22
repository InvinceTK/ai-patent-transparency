from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from articles import articles
import numpy as np
from utils import (
    array_to_str,
    get_patents_from_json,
    print_token_length,
    split_text_by_tokens,
    visualise_dict,
    visualise_reg_dict,
)
import re
import time
import json

model = SentenceTransformer("all-mpnet-base-v2")
max_tokens = model.max_seq_length
tokenizer = model.tokenizer
embeddings = {}

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

def articles_to_embeddings(articles):
    """
    Input: Articles
    Converts each article into a shape (1,768) tensor
    """
    article_dict = {}

    for title,content in articles.items():
        article_embedding =get_average_embedding(title,content,"article")
        article_embedding= article_embedding.reshape(1, -1)
        article_dict[str(title)] = article_embedding

    return article_dict

def patents_to_embeddings(patents):
    """
    Description: 
        (1) Extract abstract, claims and description from patent
        (2) Get Average of the abstract, claim and description embeddings
        
    Input: Patent object
    Output: 1 (1,768) embedding
    """
    patents_dict = {}

    for patent in patents:
        patent_id = patent.get("patent_id")
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
        average_embedding = average_embedding.reshape(1, -1)

        patents_dict[str(patent_id)] = average_embedding

    return patents_dict

def retrieve_top_k_articles(patents_dict, articles_dict, k=2):
    """
    Description: Compares each patent to every article and retrieves the top k most similar articles to a patent
    e.g Compare 1 patent -> 5 articles, k = 2, retrieve 2 articles most similar to patent

    Input: 
    patents_dict (patent_id: average_embedding), 
    articles_dict (title: average_embedding)

    Output:
    Dictionary: {
    patent_id : a1_tuple(article_title,similarity) ... ak_tuple(article_title,similarity)
    }
    """
    all_top_k = {}
   
    for patent_id, patent_embedding in patents_dict.items():
        print(patent_id)
        similarities = []
        for title, article_embedding in articles_dict.items():
            similarity = model.similarity_pairwise(patent_embedding, article_embedding)
            similarities.append((title, similarity))  # simple tuple, no dict needed

        # sort descending by similarity score, take top k
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        all_top_k[str(patent_id)] = top_k

    return all_top_k
       
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
    # print_token_length(tokenizer,f"{patent_id} {text_type} : ",text)
    text_chunked = split_text_by_tokens(tokenizer,text,max_tokens)
    for chunk in text_chunked:
        embedding = model.encode(chunk)
        sentence_embeddings.append(embedding)
    # print(f"{text_type} emb dim is {len(sentence_embeddings)}")
    sentence_embeddings= np.array(sentence_embeddings)
    average_embeddings = sentence_embeddings.mean(axis=0)
    return average_embeddings

def compute_embeddings(text_units):
    """
    Encode a list of text units and return their embeddings.

    Args:
        text_units: list[str]

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

def split_description_points(description):
    """
    Splits patent description into sections demarcated by [0001], [0002], etc.
    Each section is encoded via get_average_embedding, which handles token chunking
    for long content.

    Returns:
        List of tuples: (marker, embedding) where embedding is shape (768,)
    """
    pattern = r'(\[\d{4}\])'
    parts = re.split(pattern, description)
    sections = []

    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            embedding = get_average_embedding(marker, content, "description")
            sections.append((marker, embedding))
        i += 2

    return sections

def split_claims_points(claims_list):
    """
    Processes patent claims that are already split into a list of strings.
    Claim 1 is split across two elements (preamble + body), so we join
    consecutive non-numbered strings onto the preceding numbered claim.

    Returns:
        List of tuples: (marker, embedding) where embedding is shape (768,)
    """
    pattern = re.compile(r'^(\d+)\.')
    sections = []
    current_marker = None
    current_content = []

    for part in claims_list:
        match = pattern.match(part.strip())
        if match:
            # Save the previous claim before starting a new one
            if current_marker is not None and current_content:
                content = " ".join(current_content).strip()
                embedding = get_average_embedding(current_marker, content, "claims")
                sections.append((current_marker, embedding))

            current_marker = match.group(0)  # e.g. "1."
            # Everything after "1. " is the start of the content
            current_content = [part[match.end():].strip()]
        else:
            # Continuation of the previous claim (e.g. claim 1's body)
            if current_content is not None:
                current_content.append(part.strip())

    # Don't forget the last claim
    if current_marker is not None and current_content:
        content = " ".join(current_content).strip()
        embedding = get_average_embedding(current_marker, content, "claims")
        sections.append((current_marker, embedding))

    return sections

def compute_description_similarity(description_sections,relevant_titles,article_chunks):
    """
    Abstract: Computes similarities between description sections and article chunks

    Input: description_sections - dict containing an array of tuples 
    tuple(section_marker, section_embedding)

    Output:
    patent_results
    dict{
        section_marker : section_similarities
    }
    """

    description_results = {}

    for marker, description_emb in description_sections:
        section_similarities = {}

        for title in relevant_titles:
            for i, chunk_emb in enumerate(article_chunks[title]):
                section_similarities[f"{title}_chunk_{i}"] = float(model.similarity_pairwise(
                    description_emb.reshape(1, -1),
                    chunk_emb.reshape(1, -1),
                ))

        description_results[marker] = section_similarities
    return description_results

def compute_claim_similarity(claim_sections,relevant_titles, article_chunks):
    # need claim sections
    # {
    #     claim_marker : claim_emb
    # }

    claim_results = {}

    for marker, claim_emb in claim_sections:
        section_similarities = {}

        for title in relevant_titles:
            for i, chunk_emb in enumerate(article_chunks[title]):
                section_similarities[f"{title}_chunk_{i}"] = float(model.similarity_pairwise(
                    claim_emb.reshape(1, -1),
                    chunk_emb.reshape(1, -1),
                ))

        claim_results[marker] = section_similarities
    return claim_results

def compute_patent_article_similarities(patents, top_k_patents, articles):
    """
    Computes similarity between each patent description section and each article chunk.
    Only compares each patent against its top-k articles (from top_k_patents).

    Returns:
        {
            patent_id: {
                desc_marker: {
                    "{article_title}_chunk_{i}": similarity_score
                }
            }
        }
    """
    time_start = time.time()

    # Build a lookup so we can get the full patent object by ID
    patents_by_id = {str(p.get("patent_id")): p for p in patents}

    # Pre-compute article chunk embeddings: {title: [embedding, ...]}
    article_chunks = {} 
    for title, article_text in articles.items():
        chunks = split_text_by_tokens(tokenizer, article_text, max_tokens=80)
        article_chunks[title] = compute_embeddings(chunks)

    results = {}

    for patent_id, top_articles in top_k_patents.items():
        patent = patents_by_id[patent_id]
        description_sections = split_description_points(patent.get("description"))
        claim_sections = split_claims_points(patent.get("claims"))

        relevant_titles = [title for title, _ in top_articles]

        description_results = compute_description_similarity(description_sections,relevant_titles,article_chunks)
        claim_results = compute_claim_similarity(claim_sections,relevant_titles,article_chunks)

        results[patent_id] = {
            "descriptions" : description_results,
            "claims" : claim_results,
        }

    time_elapsed = time.time() - time_start
    return results, time_elapsed


def main():

    patents = get_patents_from_json()
    patents_dict = patents_to_embeddings(patents)
    articles_dict = articles_to_embeddings(articles)
    top_k_patents = retrieve_top_k_articles(patents_dict, articles_dict)
    results, time_elapsed = compute_patent_article_similarities(patents, top_k_patents, articles)

    with open("patent_article_similarities.json","w") as f:
        text = json.dumps(results, indent = 2)
        f.write(text)

    print(results)
    print(time_elapsed)

main()





# convert_articles_to_embeddings(articles)
# compare_article_similarity(embeddings)

# calc window size using: model.max_seq_length
# similarity = model.similarity(embeddings,embeddings)


# 384 is the max token length of sentence transformer : all-mpnet-base-v2