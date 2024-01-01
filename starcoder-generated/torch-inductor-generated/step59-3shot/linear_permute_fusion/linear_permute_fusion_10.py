
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.bn = torch.nn.BatchNorm1d(2)
    def forward(self, input):
        t1 = input.permute(0, 2, 1)
        t2 = self.linear(t1)
        t6 = t2.permute(0, 2, 1)
        t3 = self.bn(t6)
        return self.linear(t3)
# Inputs to the model
input = torch.randn(1, 4, 2)
