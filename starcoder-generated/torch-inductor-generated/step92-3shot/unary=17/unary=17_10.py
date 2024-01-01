
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = torch.nn.Linear(1, 3)
        self.dense2 = torch.nn.Linear(3, 5)
        self.dense3 = torch.nn.Linear(5, 5)
        self.dense4 = torch.nn.Linear(5, 1)
    def forward(self, x1):
        v1 = self.dense1(x1)
        v2 = torch.relu(v1)
        v3 = self.dense2(v2)
        v4 = torch.relu(v3)
        v5 = self.dense3(v4)
        v6 = torch.relu(v5)
        v7 = self.dense4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(10, 1)
