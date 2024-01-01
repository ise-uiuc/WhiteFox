
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 8)
    def forward(self, x, y):
        x = self.layers(x)
        y = torch.squeeze(self.layers(y), dim=1)
        x = (x, y)
        x = torch.cat(x, dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2)
y = torch.randn(2, 3)
