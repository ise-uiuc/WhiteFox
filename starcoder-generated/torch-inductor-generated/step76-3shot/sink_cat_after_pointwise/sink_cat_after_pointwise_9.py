
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        z = x
        for i in range(3):
            z = torch.cat([z, z, z], dim=1)
            z = z.view(x.shape[0], -1)
        return y, z
# Inputs to the model
x = torch.randn(2, 3, 4)
