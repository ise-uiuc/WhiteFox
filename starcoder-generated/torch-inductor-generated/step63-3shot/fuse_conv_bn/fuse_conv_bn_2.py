
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1000, 200)
        self.linear2 = torch.nn.Linear(1000, 500)
        self.linear3 = torch.nn.Linear(1000, 100)
        self.bn = torch.nn.BatchNorm1d(200)
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x = torch.randn(5, 1000)
