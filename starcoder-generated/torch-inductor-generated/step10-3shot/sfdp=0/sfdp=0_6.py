
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(8))
    def forward(self, x1):
        query = x1.softmax(dim=0)
        k = x1 + x1.transpose(-2, -1)
        v = x1 + x1.transpose(-2, -1)
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(query, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.argmax(dim=-1)
        output = attention_weights * x1
        return output
# Inputs to the model
x1 = torch.randn(8, 64, 64, requires_grad=False)
