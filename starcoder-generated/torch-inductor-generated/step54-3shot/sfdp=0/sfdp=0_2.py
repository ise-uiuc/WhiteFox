
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(8))
    def forward(self, x, x1):
        q = x
        k = x
        v = x
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x = torch.randn(1, 1792, 99, 131)
x1 = torch.randn(7, 11, 61)
