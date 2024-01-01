
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layer2 = nn.Linear(2, 2)
    def forward(self, x):
        x = torch.relu(self.layers(x))
        x = torch.tanh(self.layer2(x))
        return x
# Inputs to the model
x = torch.randn(2, 2)
