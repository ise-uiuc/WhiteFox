
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(10, 12)
        self.linear2 = torch.nn.Linear(12, 10)
    def forward(self, x1):
        v1 = F.relu(x1)
        v2 = self.linear1(v1)
        v3 = self.relu1(v2)
        v4 = self.linear2(v3)
        v5 = self.relu2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 10)
