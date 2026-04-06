from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from context.product_descriptions import product_descriptions
from context.idemia_articles import idemia_articles
import numpy as np
import re
import time
import json
from utils import (
    array_to_str,
    get_patents_from_json,
    print_token_length,
    split_text_by_tokens,
)


model = SentenceTransformer("all-mpnet-base-v2")
max_tokens = model.max_seq_length
tokenizer = model.tokenizer
embeddings = {}


def compare_article_similarity(embeddings):
    """
    Input : Array of embeddings (n,768)
    Prints for each article how similar it is to every other article
    """
    for title, embedding in embeddings.items():
        embedding1 = embeddings[title].reshape(1, -1)
        print(embedding1.shape)
        print(f"Calculating document similarity against {title}")

        for title, embedding in embeddings.items():
            embedding2 = embeddings[title].reshape(1, -1)
            similarity = model.similarity_pairwise(embedding1, embedding2)
            print(f"    similarity score against {title} is {similarity}")


def embed_articles(articles):
    """
    Embeds each article into a shape (1,768) tensor
    Input: array of articles

    Output:
    article_dict{
        title: average_embedding
    }
    """
    article_dict = {}

    for title, content in articles.items():
        article_embedding = get_average_embedding(title, content, "article")
        article_embedding = article_embedding.reshape(1, -1)
        article_dict[str(title)] = article_embedding

    return article_dict


def embed_patents_by_section(patents):
    """
    Maps each patent to an embedding

    Description:
        (1) Extract abstract, claims and description from patent
        (2) Get Average of the abstract, claim and description embeddings

    Input: Array of patent objects
    Output: Dict containing a patent_id as field and embedding as key
    """
    print(len(patents))
    patent_embedding_map = {}
    counter = 0
    batch_counter = 0
    total_batch_time = 0

    for patent in patents:
        time_start = time.time()
        patent_id = str(patent.get("patent_id"))

        try:
            abstract = str(patent.get("abstract") or "")
            claims = array_to_str(patent.get("claims") or [])
            description = str(patent.get("description") or "")

            abstract_embeddings = get_average_embedding(patent_id, abstract, "abstract")
            claim_embeddings = get_average_embedding(patent_id, claims, "claims")
            description_embeddings = get_average_embedding(
                patent_id, description, "description"
            )

            embeddings = np.array(
                [abstract_embeddings, claim_embeddings, description_embeddings]
            )
            average_embedding = embeddings.mean(axis=0).reshape(1, -1)

            patent_embedding_map[str(patent_id)] = average_embedding
            counter += 1
            time_elapsed = time.time() - time_start
            total_batch_time += time_elapsed
        except ValueError as e:
            print(f"Skipping patent {patent_id} — ValueError: {e}")
        except Exception as e:
            print(f"Skipping patent {patent_id} — Unexpected error: {e}")
        finally:
            if counter % 50 == 0:
                batch_counter += 1
                print(f"On the {counter}/{len(patents)} patents")
                print(f"Time elapsed on batch {batch_counter}: {total_batch_time:.2f}s")
                num_batches = len(patents) / 50
                estimated_time_remaining = (num_batches - batch_counter) * time_elapsed
                print(f"The estimated_time_remaining : {estimated_time_remaining}")
                total_batch_time = 0

    return patent_embedding_map


def map_patents_to_top_k_articles(patent_embedding_map, article_embedding_map, k=2):
    """
    Description: Compares each patent to every article and retrieves the top k most similar articles to a patent
    e.g Compare 1 patent -> 5 articles, k = 2, retrieve 2 articles most similar to patent

    Input:
    patent_embedding_map {
        patent_id: average_embedding
    }
    article_embedding_map {
        title: average_embedding
    }

    Output:
    Dictionary: {
    patent_id : a1_tuple(article_title,similarity) ... ak_tuple(article_title,similarity)
    }
    """
    time_start = time.time()
    all_top_k = {}

    for patent_id, patent_embedding in patent_embedding_map.items():
        similarities = []
        for title, article_embedding in article_embedding_map.items():
            similarity = model.similarity_pairwise(patent_embedding, article_embedding)
            similarities.append((title, similarity))  # simple tuple, no dict needed

        # sort descending by similarity score, take top k
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        all_top_k[str(patent_id)] = top_k

    time_elapsed = time.time() - time_start
    print(
        f"Took {time_elapsed} seconds to map patents to the top {k} relevant articles"
    )
    return all_top_k


def get_average_embedding(patent_id, text, text_type):
    """
    Description:
    (1) Seperate text into chunks <= max token length
    (2) Get embedding for each chunk
    (3) Take average_embedding

    Input: patentid, text, text_type
    Output: average embedding (768,)
    """
    # print_token_length(tokenizer,f"{patent_id} {text_type} : ",text)
    text_chunked = split_text_by_tokens(tokenizer, text, max_tokens)
    sentence_embeddings = np.array(
        model.encode(text_chunked)
    )  # one batched forward pass
    average_embeddings = sentence_embeddings.mean(axis=0)
    return average_embeddings


def embed_text_chunks(text_units):
    embeddings = []
    for text in text_units:
        if not text or not text.strip():
            continue
        embedding = model.encode(text)
        embeddings.append((embedding, text))  # store text alongside embedding
    return embeddings


def build_description_section_index(description):
    """
    Splits patent description into sections demarcated by [0001], [0002], etc.
    Each section is encoded via get_average_embedding, which handles token chunking
    for long content.

    Returns:
        List of tuples: (marker, embedding) where embedding is shape (768,)
    """
    pattern = r"(\[\d{4}\])"
    parts = re.split(pattern, description)
    sections = []

    i = 1
    while i < len(parts):
        marker = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            embedding = get_average_embedding(marker, content, "description")
            sections.append((marker, embedding, content))
        i += 2

    return sections


def build_claim_section_index(claims_list):
    """
    Processes patent claims that are already split into a list of strings.
    Claim 1 is split across two elements (preamble + body), so we join
    consecutive non-numbered strings onto the preceding numbered claim.

    Returns:
        List of tuples: (marker, embedding) where embedding is shape (768,)
    """
    pattern = re.compile(r"^(\d+)\.")
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
                sections.append((current_marker, embedding, content))

            current_marker = match.group(0)  # e.g. "1."
            # Everything after "1. " is the start of the content
            current_content = [part[match.end() :].strip()]
        else:
            # Continuation of the previous claim (e.g. claim 1's body)
            if current_content is not None:
                current_content.append(part.strip())

    # Don't forget the last claim
    if current_marker is not None and current_content:
        content = " ".join(current_content).strip()
        embedding = get_average_embedding(current_marker, content, "claims")
        sections.append((current_marker, embedding, content))

    return sections


def compute_description_similarity(
    description_section_index, relevant_titles, article_chunks, threshold=0.3
):
    """
    In :
    description_section_index = 
    relevant_titles = array of article titles most relevant to patent
    article_chunks = array of sections of article

    Out:
    description_sections_analysis
    [
        marker: marker
        description_content: content
        article_similarities : [
            {
                chunk_name : name_of chunk
                similarity : similarity
                chunk_content: content
            },
            ...
        ]
    ]
    """
    description_sections_analysis = []
    total_sections = len(description_section_index)
    time_start = time.time()

    for marker, description_emb, content in description_section_index:
        section_similarities = []

        for title in relevant_titles:
            for i, (chunk_emb, chunk_text) in enumerate(article_chunks[title]):
                similarity = float(
                    model.similarity_pairwise(
                        description_emb.reshape(1, -1),
                        chunk_emb.reshape(1, -1),
                    )
                )
                if similarity > threshold:
                    section_similarities.append(
                        {
                            "chunk_name": f"{title}_chunk_{i}",
                            "similarity": similarity,
                            "chunk_content": chunk_text,
                        }
                    )

        description_sections_analysis.append(
            {
                "marker": marker,
                "description_content": content,
                "article_similarities": section_similarities,
            }
        )

    total = time.time() - time_start
    print(f"compute_description_similarity — {total_sections} sections in {total:.1f}s")

    return description_sections_analysis


def compute_claim_similarity(
    claim_section_index, relevant_titles, article_chunks, threshold=0.3
):
    """
    In:
    claim_section_index : e.g .1 , .2
    relevant_titles: array titles belonging to the topk simialr articles
    article_chunks: array of sections of text from article

    Out:
    claim_sections_analysis
    [
    marker: claim_marker
    claim_content : content
    section_similarities :
        [
            {
            chunk_name : name
            similarity : similarity
            chunk_content : content
            }
            ...
        ]
    ]
    """
    claim_sections_analysis = []
    total_claims = len(claim_section_index)
    time_start = time.time()

    for marker, claim_emb, content in claim_section_index:
        section_similarities = []

        for title in relevant_titles:
            for i, (chunk_emb, chunk_text) in enumerate(article_chunks[title]):
                similarity = float(
                    model.similarity_pairwise(
                        claim_emb.reshape(1, -1),
                        chunk_emb.reshape(1, -1),
                    )
                )
                if similarity > threshold:
                    section_similarities.append(
                        {
                            "chunk_name": f"{title}_chunk_{i}",
                            "similarity": similarity,
                            "chunk_content": chunk_text,
                        }
                    )

        claim_sections_analysis.append(
            {
                "marker": marker,
                "claim_content": content,
                "article_similarities": section_similarities,
            }
        )

    total = time.time() - time_start
    print(f"compute_claim_similarity — {total_claims} claims in {total:.1f}s")

    return claim_sections_analysis


def compute_patent_article_similarities(patents, patent_to_top_articles_map, articles):
    """
    Computes similarity between each patent description section and each article chunk.
    Only compares each patent against its top-k articles (from patent_to_top_articles_map).

    In:
    patents = array of patent objects
    articles = array of articles
    patent_to_top_articles_map{
        patent_id : a1_tuple(article_title,similarity) ... ak_tuple(article_title,similarity)
    }

    Out:
    result
    [
        {
            patent_id : patent_id
            descriptions: description_sections_analyis
            claims: claim_sections_analyis
        }
        ...
    ]
    """

    # Build a lookup so we can get the full patent object by ID
    patents_by_id = {str(p.get("patent_id")): p for p in patents}

    # Pre-compute article chunk embeddings: {title: [embedding, ...]}
    article_chunks = {}
    for title, article_text in articles.items():
        chunks = split_text_by_tokens(tokenizer, article_text, max_tokens=80)
        article_chunks[title] = embed_text_chunks(chunks)

    results = []

    for patent_id, ranked_article_score_pairs in patent_to_top_articles_map.items():

        patent = patents_by_id[patent_id]

        description_section_index = build_description_section_index(
            patent.get("description") or ""
        )
        claim_section_index = build_claim_section_index(patent.get("claims") or [])

        relevant_titles = [title for title, _ in ranked_article_score_pairs]

        description_sections_analysis = compute_description_similarity(
            description_section_index, relevant_titles, article_chunks
        )

        claim_sections_analysis = compute_claim_similarity(
            claim_section_index, relevant_titles, article_chunks
        )

        results.append(
            {
                "patent_id": patent_id,
                "descriptions": description_sections_analysis,
                "claims": claim_sections_analysis,
            }
        )

    return results


def embed_product_description(product_descriptions):
    """
    Maps a product descripptipon to an embedding
    In : Array of product descriptions
    Out: Dict containing product descriptions as field and embedding as key
    """
    product_description_map = {}
    for product_name, product_description in product_descriptions.items():
        embedding = model.encode(product_description)
        embedding = embedding.reshape(1, -1)
        product_description_map[product_name] = embedding
    return product_description_map


def map_product_descriptions_to_top_k_patents(
    product_description_embedding_map, patent_embeddng_map, k=5
):
    """
    Maps indivdual products to the top k similar patents to it.

    Inputs:
    product_description_embedding_map entry = { The mq29 ghostbat ... density of 5kg : embedding }
    patent_embeddign_map {
        patent_id : embedding
        ...
        patent_id :embedding
    }
    Output:
    all_top_k_patents{
        product_name: [(patent_id, tensor([0.345])) ... (patent_id, tensor([0.345]))]
    }
    """
    all_top_k_patents = {}
    time_start = time.time()
    for (
        product_name,
        product_description_embedding,
    ) in product_description_embedding_map.items():
        print(product_name)
        similarities = []
        for patent_id, patent_embedding in patent_embeddng_map.items():
            similarity = model.similarity_pairwise(
                product_description_embedding, patent_embedding
            )
            similarities.append((patent_id, similarity))  # simple tuple, no dict needed

        # sort descending by similarity score, take top k
        top_k = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        all_top_k_patents[str(product_name)] = top_k
    time_elapsed = time.time() - time_start
    print(
        f"Took {time_elapsed} seconds to map product descriptions to the top {k} relevant patents"
    )
    return all_top_k_patents


def print_token_window_size(model):
    # 384 is the max token length of sentence transformer : all-mpnet-base-v2
    token_window_size = model.max_seq_length
    print(f"Model contains a token window size of {token_window_size}")


def main():
    try:
        chosen_product = "morphowave"
        patents = get_patents_from_json("./json/idemia_test")
        product_description_embedding_map_full = embed_product_description(
            product_descriptions
        )

        product_description_embedding_map_entry = {}
        product_description_embedding_map_entry[str(chosen_product)] = (
            product_description_embedding_map_full[chosen_product]
        )

        print("Starting embedding individual patents")
        patent_embedding_map = embed_patents_by_section(patents)
        article_embedding_map = embed_articles(idemia_articles)

        print("Starting to map product description to patents")
        product_description_to_top_patents_map = (
            map_product_descriptions_to_top_k_patents(
                product_description_embedding_map_entry, patent_embedding_map
            )
        )
        print(product_description_to_top_patents_map)

        top_patent_ids = {
            patent_id
            for top_patents in product_description_to_top_patents_map.values()
            for patent_id, _ in top_patents
        }

        filtered_patent_embedding_map = {
            patent_id: emb
            for patent_id, emb in patent_embedding_map.items()
            if patent_id in top_patent_ids
        }

        patent_to_top_articles_map = map_patents_to_top_k_articles(
            filtered_patent_embedding_map, article_embedding_map
        )

        filtered_patents = [
            p for p in patents if str(p.get("patent_id")) in top_patent_ids
        ]

        results = compute_patent_article_similarities(
            filtered_patents, patent_to_top_articles_map, idemia_articles
        )

        with open("patent_article_similarities.json", "w") as f:
            json.dump(results, f, indent=2)

        print("✅ Results saved to patent_article_similarities.json")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


def main():
    try:
        chosen_product = "morphowave"
        patents = get_patents_from_json("./json/idemia_test")
        product_description_embedding_map_full = embed_product_description(
            product_descriptions
        )

        product_description_embedding_map_entry = {}
        product_description_embedding_map_entry[str(chosen_product)] = (
            product_description_embedding_map_full[chosen_product]
        )

        print("Starting embedding individual patents")
        patent_embedding_map = embed_patents_by_section(patents)
        article_embedding_map = embed_articles(idemia_articles)

        print("Starting to map product description to patents")
        product_description_to_top_patents_map = (
            map_product_descriptions_to_top_k_patents(
                product_description_embedding_map_entry, patent_embedding_map
            )
        )
        print(product_description_to_top_patents_map)

        top_patent_ids = {
            patent_id
            for top_patents in product_description_to_top_patents_map.values()
            for patent_id, _ in top_patents
        }

        filtered_patent_embedding_map = {
            patent_id: emb
            for patent_id, emb in patent_embedding_map.items()
            if patent_id in top_patent_ids
        }

        patent_to_top_articles_map = map_patents_to_top_k_articles(
            filtered_patent_embedding_map, article_embedding_map
        )

        filtered_patents = [
            p for p in patents if str(p.get("patent_id")) in top_patent_ids
        ]

        results = compute_patent_article_similarities(
            filtered_patents, patent_to_top_articles_map, idemia_articles
        )

        with open("patent_article_similarities.json", "w") as f:
            json.dump(results, f, indent=2)

        print("✅ Results saved to patent_article_similarities.json")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


main()


# {'morphowave': [('EP3931005', tensor([nan], dtype=torch.float64)), ('EP4383208', tensor([0.3700])), ('EP4202857', tensor([0.3666])), ('WO2024260672', tensor([0.3628])), ('EP4660873', tensor([0.3624]))]}
