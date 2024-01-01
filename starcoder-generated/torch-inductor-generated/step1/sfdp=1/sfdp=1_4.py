
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.kvp_proj = torch.nn.Linear(d_model, d_model)
        self.output_projection = torch.nn.Linear(d_model, d_model)
 
    def forward(self, query, key, value, dropout_p=0.2):
        v1 = self.kvp_proj(query)
        v2 = self.kvp_proj(key)
        v3 = self.kvp_proj(value)
        v4 = v1.matmul(v2.transpose(-2, -1).float()) / (self.d_model ** 0.5)
        v5 = F.softmax(v4, -1).to(v4)
        v6 = F.dropout(v5, p=dropout_p)
        v7 = v6.matmul(v3).to(v4)
        return self.output_projection(v7)

# Initializing the model
m = MultiHeadedAttention(64, 4)

# Inputs to the model
query = torch.randn(1, 4, 64)
key = torch.randn(1, 4, 64)
value = torch.randn(1, 4, 64)
