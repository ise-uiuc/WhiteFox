
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query_embedding = torch.nn.Embedding(num_embeddings=768, embedding_dim=32)
        self.key_embedding = torch.nn.Embedding(num_embeddings=768, embedding_dim=32)
        self.value_embedding = torch.nn.Embedding(num_embeddings=768, embedding_dim=32)
        self.attn_dropout = torch.nn.Dropout(0.3)
 
    def forward(self, x1, x2, x3):
        w1 = self.query_embedding(x1)
        w2 = self.key_embedding[x2]
        w = w1 @ w2 / math.sqrt(w1.size(-1))
        m2 = torch.full((1, 10, 768), float("-inf"), dtype=torch.float32).to('cuda')
        v4 = torch.cat([w, m2], dim=1)
        v5 = torch.softmax(0.3 * v4, dim=-1)
        v6 = torch.dropout(v5, 0.3, True)
        v7 = self.value_embedding(x3)
        v8 = v5 @ v7
        return v8

# Initializing the model
m = Model()
m.to('cuda')

# Inputs to the model
x1 = torch.randint(low=0, high=768, size=(5,), device='cuda')
x2 = torch.randint(low=0, high=768, size=(6, 32), device='cuda')
x3 = torch.randint(low=0, high=768, size=(2, 10), device='cuda')
