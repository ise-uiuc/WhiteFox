
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.zeros_like(x)
        x = torch.add(x, 1)
        x = x.detach()
        return x
# Inputs to the model
x = torch.randn(2, 2)
