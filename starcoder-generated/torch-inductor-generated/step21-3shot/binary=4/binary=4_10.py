
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10, bias=False)
        self.linear2 = torch.nn.Linear(10, 10, bias=False)
        self.bn = torch.nn.BatchNorm1d(10)
 
    def forward(self, x0):
        v0 = torch.add(self.linear1(x0).mean(dim=-1, keepdim=True), self.linear2(x0))
        t0 = v0.transpose(-1, -2)
        v1 = self.bn(t0)
        v2 = v1.transpose(-1, -2)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x0 = torch.randn(5, 1, 10)
