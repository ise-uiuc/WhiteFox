
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = x
    def forward(self, x):
        a = torch.cat([x, self.x], dim=1)
        b = torch.cat([x, x], dim=0)
        z = torch.tanh(b)
        w = torch.cat([a, z], dim=1)
        z = torch.relu(w)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
