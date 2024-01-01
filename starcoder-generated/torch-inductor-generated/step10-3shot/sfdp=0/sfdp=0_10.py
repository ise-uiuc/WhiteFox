
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(8))
    def forward(self, x2):
        q = x2
        k = x2
        v = x2
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = scaled_dot_product.matmul(v)
        return output
# Inputs to the model
x2 = torch.randn(1, 8, 64)
