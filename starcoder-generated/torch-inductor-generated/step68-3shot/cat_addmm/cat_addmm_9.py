
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 10)
