
class Model(torch.nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super().__init__()
        self.query_linear = torch.nn.Linear(hidden_dim, d_model)
        self.key_linear = torch.nn.Linear(hidden_dim, d_model)
 
    def attention(self, query, key, value, mask=None):
        k = self.key_linear(key)
        q = self.query_linear(query)
        qk = q @ k.transpose(-2, -1) # Compute the dot product of the query and key, and scale it
        qk = qk / math.sqrt(q.size(-1))
        if mask is not None:
            qk += mask # Add the attention mask to the scaled dot product
        attn_weight = torch.softmax(qk, dim=-1) # Apply softmax to the result
        output = attn_weight @ value # Compute the dot product of the attention weights and the value
        return output
 
    def forward(self, x1, x2):
        v1 = self.attention(x1,x2,x2)
        return v1

# Initializing the model
d_model = 512
hidden_dim = 2048
m = Model(d_model, 8, hidden_dim)

# Inputs to the model
x1 = torch.randn(1, 8, 64, 2048)
x2 = torch.randn(1, 8, 256, 2048)
