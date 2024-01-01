
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers0 = nn.Linear(8, 16)
        self.layers1 = nn.Linear(16, 2)
    def forward(self, x):
        x = self.layers0(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = torch.relu(x)
        x = self.layers1(x)
        return x
# Inputs to the model
x = torch.randn(2, 8)
