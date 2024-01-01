
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 1)
    def forward(self, x):
        y0, y1 = x.chunk(2, dim=1)
        y2 = y0 - y1
        z = torch.cat([y0, y1, y2], dim = 1)
        z = self.linear(z)
        w, x, y = z.chunk(3, dim=1)
        w = w + x
        return torch.relu(w + y)
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
