
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key = torch.nn.Parameter(torch.randn(3073, 2137, 2371))
    def forward(self, x1):
        q = x1
        k = x1
        inv_scale = math.sqrt(k.size(1))
        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / inv_scale
        scaled_dot_product.softmax(dim=-1).matmul(k).softmax(dim=-1)
        scaled_dot_product.softmax(dim=-1)
        return scaled_dot_product.softmax(dim=-1)
# Inputs to the model
x1 = torch.randn(23, 3, 1522, 29)
