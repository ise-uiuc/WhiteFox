
class MyModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, q, k, v, mask=None):
        q = q / (dim ** -0.5)

        dot_product = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            dot_product.mul_(mask)

        attention = torch.softmax(dot_product, dim=-1)
        output = torch.matmul(attention, v)

        return output

# Initialize the model
model = MyModel(4096)

# Inputs to the model
q = torch.randn(100, 1024)
k = torch.randn(100, 4096)
v = torch.randn(100, 4096)
mask = torch.zeros(100, 4096, 4096)
output = model(q, k, v, mask)

