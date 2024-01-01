
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        if x.ndim == 2:
            x = x.flatten()
            x = x.unsqueeze(0)
            x = [x]
        x = torch.cat(x, dim=0)
        return x
# Inputs to the model
x = torch.randn(1, 2)
x1 = torch.randn(2, 2)
x2 = torch.randn(3, 2)
