from sentence_transformers import SentenceTransformer

class SentenceEmbedding(object):
    """docstring for SentenceEmbedding"""

    def __init__(self,path='paraphrase-multilingual-mpnet-base-v2'):
        self.model = self.loadModel(path)

    def loadModel(self,path):
        print(f'Loading model from {path}')
        model = SentenceTransformer(path)
        print('Model Loaded.')
        return model
    def embed(self, input_):
        embedding = self.model.encode(input_,convert_to_tensor=True)
        question_embeddings = embedding.cpu().numpy().tolist()
        return question_embeddings

obj = SentenceEmbedding()
def getEmbeddings(data):
    return obj.embed(data)
    