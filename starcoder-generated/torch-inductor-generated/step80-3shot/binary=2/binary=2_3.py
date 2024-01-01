
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8,16)
        self.linear2 = torch.nn.Linear(16,8)
        self.linear3 = torch.nn.Linear(8,8)
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = F.leaky_relu(x - 3.1)
        v3 = self.linear2(v1)
        v4 = v3 - 5000.0
        v5 = self.linear3(v2)
        v6 = v5 - 10.0
        return v6
# Inputs to the model
x = torch.randn(1, 8)
