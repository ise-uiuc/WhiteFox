
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input = torch.nn.Parameter(torch.randn(3, 64, 64))
    def forward(self, x1):
        q = x1
        k = x1
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
