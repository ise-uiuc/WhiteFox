
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 8)
        self.linear2 = torch.nn.Linear(8, 8)
        self.linear3 = torch.nn.Linear(8, 1)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.linear3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(4, 1)
