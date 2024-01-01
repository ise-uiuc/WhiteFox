
class Model(torch.nn.Module):
    def __init__(self, hidden_size, dropout_p=0.05, scale_factor=1 / (hidden_size ** 0.4)):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.proj = Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_p)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = scale_factor

    def forward(self, queries, keys, values):
        # Project queries and keys. Apply dropout
        query_proj = self.proj(queries)
        key_proj = self.proj(keys)
        queries_and_keys = self.dropout(query_proj.unsqueeze(-2) + key_proj.unsqueeze(-3))
        # Compute scaled dot products
        attention_weights = self.scale_factor * (queries_and_keys.sum(-1))
        # Apply softmax
        attention_weights = self.softmax(attention_weights).unsqueeze(-1)
        scores = torch.matmul(attention_weights, values).squeeze(-1)
        return scores

# Initializing the model
m = Model(hidden_size=3136)
m(x1, x2, x2)
# Inputs to the model
x1 = torch.randn(2, 1284, 3136) # (seq len, batch, hidden size)
x2 = torch.randn(2, 1284, 3136) # (seq len, batch, hidden size)
x3 = torch.randn(2, 1284, 3136) # (seq len, batch, hidden size)
