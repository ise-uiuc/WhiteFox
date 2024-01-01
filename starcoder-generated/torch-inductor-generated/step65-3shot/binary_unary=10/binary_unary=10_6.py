
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        embedding_size = 11
        num_embeddings = 20
        self.emb = torch.nn.EmbeddingBag(num_embeddings, embedding_size, mode="sum", sparse=True) # EmbeddingBag
        self.embedding_size = embedding_size      # A scalar indicating the number of embeddings
        self.num_embeddings = num_embeddings    # The maximum number of items in the embedding matrix
 
    def forward(self, x1):
        x = self.emb(x1)
        x = relu(x)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.full((1, 16), 3, dtype=torch.int64)
print(x1)
