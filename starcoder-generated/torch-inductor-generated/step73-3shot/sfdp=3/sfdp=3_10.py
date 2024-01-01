
class Model(torch.nn.Module):
    def __init__(self, hidden_size, num_heads, num_queries, num_keys, max_len, dropout_p=0.1):
        super().__init__()
        self.dropout_p = dropout_p
        self.dot_product_attention = torch.nn.MultiheadAttention(hidden_size, num_heads)
        self.query_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.key_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.value_projection = torch.nn.Linear(hidden_size, hidden_size)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, queries, keys, values):
        x1 = self.query_projection(queries)
        x2 = self.key_projection(keys)
        x3 = self.value_projection(values)
        x4 = self.dot_product_attention(x1, x2, x3, padding_mask=None, need_weights=False)[0]
        x5 = self.dropout(x4)
        x6 = self.softmax(x5)
        return x6

# Initializing the model
m = Model(hidden_size=15, num_heads=3, num_queries=4, num_keys=5, max_len=7)

# Inputs to the model
q = torch.randn(2, 3, 3, 15)
k = torch.randn(2, 5, 3, 15)
v = torch.randn(2, 5, 3, 15)
