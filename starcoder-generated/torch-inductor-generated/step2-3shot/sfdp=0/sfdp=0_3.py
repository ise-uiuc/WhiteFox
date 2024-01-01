
class model(torch.nn.Module):
    def __init__(self, dim_q, dim_k):
        super().__init__()
        self.to_q = torch.nn.Linear(dim_q, dim_q)
        self.to_k = torch.nn.Linear(dim_k, dim_k)
    def forward(self, query, key):
        q = self.to_q(query)
        k = self.to_k(key)
        dot_score = q @ k.transpose(-2, -1) / math.sqrt(dim_k)
        attention_weight = F.softmax(dot_score, -1)
        output = attention_weight @ value
        return output

# Initializing the model
m = LayerNorm(3, 5)

# Inputs to the model
query = torch.randn(1, 4, 3)
key = torch.randn(1, 2, 5)
value = torch.randn(1, 2, 3)
