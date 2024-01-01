
class MultiHeadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Linear(3, 6)
        self.key = torch.nn.Linear(4, 8)
        self.value = torch.nn.Linear(5, 10)
 
    def forward(self, query, key, value, scale_factor=1.0, dropout_p=0.5):
        q, k, v = self.query(query), self.key(key), self.value(value)
        attention_output = softmax_with_mask(
            torch.matmul(q, k.transpose(-1, -2)),
            scale_factor=scale_factor,
            dropout_p=dropout_p)
        out = attention_output.matmul(v)
        return out

# Initializing and testing the model
m = MultiHeadAttention()
query = torch.randn(10, 3)
key = torch.randn(10, 4)
value = torch.randn(10, 5)
