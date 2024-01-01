
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(8))
        self.key = torch.nn.Parameter(torch.randn(8, 64))
    def forward(self, x2):
        q = self.query
        k = self.key
        v = x2
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x2 = torch.randn(1, 8, 64, 64)
