
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 6)
    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 2, 3)
        return x
# Inputs to the model
x = torch.randn(6, 3)
