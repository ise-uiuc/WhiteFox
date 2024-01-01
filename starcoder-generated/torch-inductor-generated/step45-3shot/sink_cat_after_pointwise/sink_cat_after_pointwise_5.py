
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.sigmoid
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        z = self.a(y)
        w = z.view(z.shape[0], -1)
        x = w.neg()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
