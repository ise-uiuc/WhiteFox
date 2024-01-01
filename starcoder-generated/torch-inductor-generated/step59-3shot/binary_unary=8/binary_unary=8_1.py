
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 64)
        self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 32)
        self.linear4 = torch.nn.Linear(32, 16)
    def forward(self, x1):
        v1 = self.linear1(x1.reshape(64))
        v2 = self.linear2(x1.reshape(32, 4))
        v3 = self.linear3(x1.reshape(2, 8, 2))
        v4 = self.linear4(x1.reshape(1, 2, 2, 2))
        v5 = v1 + v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
# Shape: [64 x 1]
x1 = torch.randn(1, 64)
# Shape: [32 x 4]
x2 = torch.randn(1, 32, 4)
# Shape: [2 x 8 x 2]
x3 = torch.randn(1, 2, 8, 2)
# Shape: [1 x 2 x 2 x 2]
x4 = torch.randn(1, 1, 2, 2, 2)
