
class Model(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.linear_query = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_key = torch.nn.Linear(embed_dim, embed_dim)
        self.linear_value = torch.nn.Linear(embed_dim, embed_dim)
        self.dropout = torch.nn.Dropout(0.1)
 
    def forward(self, query, key, value, dropout_p):
        query = self.linear_query(query)
        key = self.linear_key(key)
        value = self.linear_value(value)
        shape_q = (query.size(0), -1, self.num_heads, query.size(-1))
        shape_k = (key.size(0), -1, self.num_heads, key.size(-1))
        shape_v = (value.size(0), -1, self.num_heads, value.size(-1))
        q = query.view(*shape_q)
        k = key.view(*shape_k)
        v = value.view(*shape_v)
        q = q.transpose(-3, -2)
        k = k.transpose(-3, -2)
        v = v.transpose(-3, -2)
        qk = torch.matmul(q, k.transpose(-2, -1))
        scale = k.size(-1) ** -0.5
        qk = qk * scale
        softmax_qk = torch.nn.functional.softmax(qk, dim=-2)
        softmax_qk = self.dropout(softmax_qk)
        output = torch.matmul(softmax_qk, v)
        shape_o = (output.size(0), output.size(1), -1)
        output = output.transpose(-3, -2).contiguous().view(*shape_o)
        return output

# Initializing the model
m = Model(256, 8)

# Inputs to the model
query = torch.randn(1, 42, 256)
key = torch.randn(1, 40, 256)
value = torch.randn(1, 40, 256)
dropout_p = 0.3
