
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(28, 16)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 6)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.relu(v1)
        v3 = self.linear2(v2)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 28)
