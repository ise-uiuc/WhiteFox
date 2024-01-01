
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(1, 2, 3, 4))
        self.query = torch.nn.Parameter(torch.randn(1, 2, 3, 4))
        self.value = torch.nn.Parameter(torch.randn(1, 2, 3, 4))
    def forward(self, x1):
        q = x1
        k = self.key
        v = self.value
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(v)
        return output
# Inputs to the model
x1 = torch.randn(2, 3, 4, 5)
