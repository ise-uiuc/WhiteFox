
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.cat
    def forward(self, x):
        x = torch.randn(2, 4)
        y = torch.randn(2, 4)
        x = self.layers([x, y], dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2)
