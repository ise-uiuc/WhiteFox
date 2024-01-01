
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(42, 60, 26, 98))
    def forward(self, x1):
        q = x1.view(24, 81, 99, 39)
        k = x1.view(6, 9, 43, 46)
        v = x1.view(58, 82, 26, 30)
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(92, 81, 22, 82)
