
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        x = x * y * z
        return x
# Inputs to the model
x = torch.ones(2, 2)
y = torch.ones(2, 2)
z = torch.ones(2, 2)
