
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_query = torch.nn.Embedding()
        self.embed_key = torch.nn.Embedding()
        self.embed_value = torch.nn.Embedding()
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout=0.5)
 
    def forward(self, query, key, value):
        query = self.embed_query(query)
        key = self.embed_key(key)
        value = self.embed_value(value)
        att = self.scaled_dot_product_attention(query, key, value)
        return att

# Initializing the model
m = Model()

# Inputs to the model
query = torch.tensor([[3, 3]])
key = torch.tensor([[4, 3, 4]])
value = torch.tensor([[4, 3, 4, 4, 3, 4],
                      [1, 2, 1, 1, 1, 2],
                      [5, 4, 6, 6, 5, 7]])
dropout_p = 0.5
scale_factor = math.sqrt(query.size(-1) * key.size(-1))
