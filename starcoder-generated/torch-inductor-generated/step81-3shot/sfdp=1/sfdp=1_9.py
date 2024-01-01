
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(16, 2, dim=3)

    def forward(self, query, key, value):
        attention_output = self.attention(query, key, value)
        return attention_output

# Inputs to the model
query, key, value = torch.randn(2, 2, 2, 16, 32, 32), torch.randn(2, 2, 2, 1, 32, 32), torch.randn(2, 2, 2, 16, 32, 32)
