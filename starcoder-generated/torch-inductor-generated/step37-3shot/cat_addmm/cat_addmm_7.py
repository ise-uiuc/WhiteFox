
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(100, 2)
        self.layers2 = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        return x
# Inputs to the model
x = torch.randn(8, 100)
