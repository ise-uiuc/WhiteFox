
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 3)
    def forward(self, x):
        x = torch.concat((x, x), dim=-1)
        x = self.layers(x)
        y = torch.tensor([1, 2, 3])
        return torch.stack(x + y, dim=-1)
# Inputs to the model
x = torch.randn(2, 5)
