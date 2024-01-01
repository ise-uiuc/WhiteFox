
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.bias = nn.Parameter(torch.randn(2, 1))
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x, x + self.bias], dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2)
