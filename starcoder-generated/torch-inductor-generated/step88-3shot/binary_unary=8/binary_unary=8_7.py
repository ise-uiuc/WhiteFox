
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(6, 1, bias=True)
        self.linear2 = torch.nn.Linear(6, 1, bias=False)
        self.linear3 = torch.nn.Linear(6, 1, bias=False)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = self.linear2(x1)
        v3 = self.linear3(x1)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6)
