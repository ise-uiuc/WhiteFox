
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(3, 2, 5))
        self.U = torch.nn.Parameter(torch.randn(3, 6, 2))
    def forward(self, x1):
        w = self.W
        u = self.U
        q = x1
        k = x1
        v = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(1, 3, 128, 192)
