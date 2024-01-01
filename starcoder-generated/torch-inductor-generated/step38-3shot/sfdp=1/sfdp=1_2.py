
class Model(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, dropout_p):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = torch.nn.Parameter(torch.ones(1, num_heads, 1, 1) * (embedding_dim // num_heads))
        self.dropout = torch.nn.Dropout(dropout_p)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        qk = torch.matmul(query, key.transpose(-2, -1))
        inv_scale_factor = self.temperature.reciprocal()
        scaled_qk = qk * inv_scale_factor
        softmax_qk = self.softmax(scaled_qk)
        dropout_qk = self.dropout(softmax_qk)
        output = torch.matmul(dropout_qk, value)
        return output


# Initializing the model
m = Model(num_embeddings=10, embedding_dim=32, num_heads=8, dropout_p=0.1)

# Inputs to the model
query = torch.randn(1, 8, 5, 32)
key = torch.randn(1, 8, 7, 32)
value = torch.randn(1, 8, 7, 32)
