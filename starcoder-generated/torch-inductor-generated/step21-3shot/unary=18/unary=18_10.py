
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=196, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=16)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.fc2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 196)
