
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(16, 16))
    def forward(self, x1):
        q = x1
        k = torch.cat([x.unsqueeze(1) for x in self.key.transpose(1, 0)], dim=1)
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        attention_weights = scaled_dot_product.softmax(dim=-1)
        output = attention_weights.matmul(self.key.transpose(0, 1))
        return output
# Inputs to the model
x1 = torch.randn(32, 16, 64, 64)
