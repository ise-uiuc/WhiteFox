
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layers(x))
        x = torch.stack((x, x), dim=0).flatten('A', 0, 1)
        return x
# Inputs to the model
x = torch.randn(2, 3)
