
class Model(torch.nn.Module):
    def __init__(self, num_queries, num_keys, num_values, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.num_keys = num_keys
        self.num_values = num_values
        self.query_projection = torch.nn.Linear(num_queries, num_queries)
        self.key_projection = torch.nn.Linear(num_keys, num_keys)
        self.value_projection = torch.nn.Linear(num_values, num_values)

    def scaled_dot_product_attention(self, query, key, value, attn_mask):
        s = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        s += attn_mask
        attn_weights = torch.softmax(s, dim=-1)
        return attn_weight @ value

    def forward(self, query, key, value, attn_mask):
        query = self.query_projection(query)
        key = self.key_projection(key)
        value = self.value_projection(value)
        return self.scaled_dot_product_attention(query, key, value, attn_mask)


# Initializing the model
m = Model(10, 20, 30, 2)

# Inputs to the model
query = torch.randn(4, 10)
key = torch.randn(4, 20)
value = torch.randn(4, 30)
attn_mask = torch.randn(4, 1, 1, 20)
