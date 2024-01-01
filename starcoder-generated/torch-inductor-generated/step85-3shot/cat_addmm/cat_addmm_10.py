
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3, 3)
    def forward(self, x):
        return self.layers(x)
# Inputs to the model
x = torch.randn(2, 2)
