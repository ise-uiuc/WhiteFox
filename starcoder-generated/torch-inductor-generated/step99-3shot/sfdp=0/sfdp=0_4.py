
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(84, 33, 81, 100))
        self.query = torch.nn.Parameter(torch.randn(64, 100, 4, 7))
    def forward(self, x1):
        q = self.query
        k = self.key
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(45, 12, 5, 69)
