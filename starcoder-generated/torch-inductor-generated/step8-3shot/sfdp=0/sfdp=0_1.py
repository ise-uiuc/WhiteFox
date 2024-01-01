
class QueryKeyAttention(torch.nn.Module):
    def __init__(self, n_state):
        super().__init__()
        self.query = torch.nn.Parameter(torch.FloatTensor(n_state))
        self.key = torch.nn.Parameter(torch.FloatTensor(n_state))
        self.value = None
        self.inv_scale = 1.0 / math.sqrt(n_state)

    def set_value(self, value):
        self.value = value

    def forward(self, query, key):
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / self.inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(self.value)
        return output

# Initializing the model
query = torch.randn(1, 64, 256)
key = torch.randn(1, 64, 1024)
