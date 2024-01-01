
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
    def forward(self, x):
        z = torch.cat([x, x, x], dim=1)
        x = z.tanh()
        x = torch.cat([x, x], dim=3)
        x = torch.cat([x, x], dim=3)
        return x.relu()
# Inputs to the model
x = torch.randn(2, 4, 3)
