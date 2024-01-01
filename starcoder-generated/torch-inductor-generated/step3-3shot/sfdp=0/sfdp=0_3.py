
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=32, num_heads=2):
        super().__init__()

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads

        # Use torch.nn.Parameter to prevent the gradients of the weight matrices being destroyed by the method assign_embeddings.
        self.queries = torch.nn.Linear(d_model, d_model)
        self.keys = torch.nn.Linear(d_model, d_model)
        self.values = torch.nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        # split heads
        queries = self.queries(queries).view(1, -1, self.num_heads, self.head_dim).split(1, dim=1)
        keys = self.keys(keys).view(-1, 1, self.num_heads, self.head_dim).split(1, dim=0)
        values = self.values(values).view(-1, 1, self.num_heads, self.head_dim).split(1, dim=0)

        query_vectors = torch.cat([split.squeeze(1) for split in queries], dim=1)
        key_vectors = torch.cat([split.squeeze(1) for split in keys], dim=1)
        value_vectors = torch.cat([split.squeeze(1) for split in values], dim=1)

        # obtain dot product score and scale
        attention_weights = torch.matmul(query_vectors, key_vectors.transpose(-2, -1)).softmax(-1)
        scaled_attention = (attention_weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))).matmul(value_vectors)
        return scaled_attention.view(1, -1, self.d_model)

# Initializing the model
m = MultiHeadAttention()

# Inputs to the model (both the input query vector and the input key and value vectors are the same)
x1 = torch.randn(1, 32, 32)
x2 = torch.randn(32, 32)
x3 = torch.randn(32, 32)
