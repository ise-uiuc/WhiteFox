
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x2 = torch.nn.Parameter(torch.randn(9, 88, 79, 53))
        self.x3 = torch.nn.Parameter(torch.randn(453, 63, 52, 84))
    def forward(self, x1):
        q = x1
        k = x1
        v = self.x2
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(26, 52, 10, 3)
