import google.generativeai as genai #pip install google-generativeai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def embed_chunks(chunks):
    embedding =[]
    for i,chunk in enumerate(chunks):
        #skip empty chunks
        if not chunk.strip():
            embedding.append(np.zeros(768)) #or appropriate embedding size
            continue
        
        #limited chunk length
        chunk = chunk[:3000]      
        try:
            res=genai.embed_content(
                model='model/embedding-001',
                content = chunk,
                task_type='retrieval_document'
            )
            embedding.append(res['embedding'])
        except Exception as e:
            print(f'embedding failed for chunk {i}:{e}')
            embedding.append(np.zeros(768)) #fallback zero vector
            
        
        return np.array(embedding)
    
def embed_query(query):
    if not query.strip():
        return np.zeros(768) #fallback zero vector   
    try:
        res=genai.embed_content(
            model='model/embedding-001',
            content = query,
            task_type='retrieval_document'
        )
        return res['embedding']
    except Exception as e:
        print(f'Embedding failed for query:{e}')
        return np.zeros(768)

def retrieve_relevant_chunks(query,chunks,chunk_embedding,top_k=5):
    query_embed = embed_query(query)
    similarities = cosine_similarity([query_embed],chunk_embedding)
    top_indices = similarities.argsort()[-top_k:][::-1]
    print(top_indices)
    return [chunks[i] for i in top_indices[0]]
        