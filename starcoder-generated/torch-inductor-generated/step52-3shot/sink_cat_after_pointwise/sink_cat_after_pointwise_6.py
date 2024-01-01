
class Reshape(torch.nn.Module):
    def __init__(self, t):
        super().__init__()
        self.t = t
    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = self.t(x)
        z = y.tanh()
        z = torch.cat([z, z, z], dim=1)
        q = z.view(z.shape[0], -1)
        q = q.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
