
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(256, 128)
        self.linear_2 = torch.nn.Linear(128, 64)
        self.linear_3 = torch.nn.Linear(64, 32)
    def forward(self, x1):
        v1 = self.linear_1(x1)
        v2 = F.relu(v1)
        v3 = self.linear_2(v2)
        v4 = F.relu(v3)
        v5 = self.linear_3(v4)
        v6 = F.relu(v5)
        v7 = v1 * v5
        v8 = v2 + v4 + v6 + v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 256)
