
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 3)
    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = x + x
        x = torch.transpose(x, 1, 2)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(4, 4)
