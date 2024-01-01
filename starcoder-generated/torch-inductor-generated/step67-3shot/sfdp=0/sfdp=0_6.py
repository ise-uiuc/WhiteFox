
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(15, 24, 58, 80))
    def forward(self, x1):
        q = x1
        k = x1
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model (1)
x1 = torch.randn(1, 17, 72, 31)
# Inputs to the model (2)
x2 = torch.randn(38, 5, 84, 90)
