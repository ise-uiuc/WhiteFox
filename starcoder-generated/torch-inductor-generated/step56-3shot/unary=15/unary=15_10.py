
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3)
