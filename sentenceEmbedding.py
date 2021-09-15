from sentence_transformers import SentenceTransformer

class SentenceEmbedding(object):
    """docstring for SentenceEmbedding"""

    def __init__(self,path='paraphrase-multilingual-mpnet-base-v2'):
        self.model = self.loadModel(path)

    def loadModel(self,path):
        if self.model is None:
            print('Loading Tensorflow model....')
            model = SentenceTransformer(path)
            print('Model Loaded.')
            return model
    def embed(self, input_):
        embedding = model.encode(input_,convert_to_tensor=True)
        question_embeddings = embedding.cpu().numpy().tolist()
        return question_embeddings

obj = SentenceEmbedding()
def getEmbeddings(data):
    obj.embed(data)
    return vector