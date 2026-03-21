import gensim
import gensim.downloader as api

dataset = api.load("text8")
data = [i for i in dataset]

def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])
    
training_data = list(tagged_document(data))
model = gensim.models.doc2vec.Doc2Vec(vector_size = 40, min_count=2, epochs=30)
model.build_vocab(training_data)
model.train(training_data, total_examples=model.corpus_count, epochs=model.epochs)


vectors = [model.infer_vector([word for word in sent]).reshape(1,-1) for sent in sentences]

similarity = []
for i in range(len(sentences)):
    row = []
    for j in range(len(sentences)):
        row.append(cosine_similarity(vectors[i],vectors[j])[0][0])
    similarity.append(row)
      
create_heatmap(similarity)