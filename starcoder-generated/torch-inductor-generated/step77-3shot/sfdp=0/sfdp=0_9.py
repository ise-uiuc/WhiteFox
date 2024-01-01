
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(45, 99))
    def forward(self, x1):
        q = x1
        k = self.key
        v = self.key
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(86, 18, 20, 3, 39)
x2 = torch.randn(47, 99, 16, 83, 34)
