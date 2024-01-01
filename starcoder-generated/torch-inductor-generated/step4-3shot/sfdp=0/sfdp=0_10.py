
class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(32, query_dim, bias=False)
        self.key = torch.nn.Linear(48, key_dim, bias=False)
        self.value = torch.nn.Linear(16, value_dim, bias=False)
 
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(query_dim)
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output

# Initializing the model
m = AttentionModel()

# Inputs to the model
x1 = torch.randn(batch_size, 32)
x2 = torch.randn(batch_size, 48)
