
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).view(12, -1)
        y = x.tanh()
        z = y.view(x.shape[0], x.shape[1], x.shape[2], x.shape[4])
        x = x.view(z.shape[0] + x.shape[0], z.shape[1], z.shape[2], z.shape[3])
        x = torch.cat([x, z], dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
