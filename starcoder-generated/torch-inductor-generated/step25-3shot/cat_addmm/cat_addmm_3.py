
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x, z):
        x = self.layers(x)
        z = self.layers(z)
        t1 = (x.transpose(-1, -2) @ z).squeeze(-1)
        x = torch.stack((t1, t1, t1, t1), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
z = torch.randn(2, 2)
