
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        z = x.transpose(dim0=1, dim1=2).relu().view(x.shape[0], x.shape[1], -1).transpose(dim0=1, dim1=2)
        y = torch.relu(z)
        return y + y
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 5)
z = torch.randn(2, 3, 4)
