
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_dim = args.embedding_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
 
    def forward(self, inputs):
        v1 = self.embedding(inputs)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
