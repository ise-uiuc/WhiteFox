
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        z = x
        for i in range(4):
            y = z * x
            y = torch.cat((y, y, y), dim=1)
            y = y.view(z.shape[0], -1) + y.view(-1, z.shape[0])
            y = y.relu().transpose(0, 1)
            z = y * z
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
