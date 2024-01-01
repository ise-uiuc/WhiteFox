
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(288, 1024, bias=True)
        self.linear2 = torch.nn.Linear(1024, 1024, bias=True)
        self.linear3 = torch.nn.Linear(1024, 16, bias=True)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = torch.relu(v1)
        v3 = self.linear2(v2)
        v4 = torch.relu(v3)
        v5 = self.linear3(v4)
        v6 = torch.view(v5, (-1, 16, 1, 4))
        return v6
# Inputs to the model
x1 = torch.randn(1, 288)
