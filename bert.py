from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence1 = "I like coding in Python."
sentence2 = "Python is my favourite programming language."

tokens1 = tokenizer.tokenize(sentence1)
tokens2 = tokenizer.tokenize(sentence2)
tokens1 = ['[CLS]'] + tokens1 + ['[SEP]'] 
tokens2 = ['[CLS]'] + tokens2 + ['[SEP]']

# batch size 0
input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0) 
input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0) 

with torch.no_grad():
    outputs1 = model(input_ids1)
    outputs2 = model(input_ids2)

    embeddings1 = outputs1.last_hidden_state[:,0,:]
    embeddings2 = outputs2.last_hidden_state[:,0,:]

similarity_score = cosine_similarity(embeddings1,embeddings2)
print("Similarity Score: ",similarity_score)
