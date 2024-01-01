
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(2, 2)
        self.batchnorm = torch.nn.BatchNorm1d(num_features=1)
    def forward(self, x1):
        m1 = self.linear1(x1)
        v1 = self.linear2(m1)
        o1 = m1 + v1
        o2 = self.batchnorm(o1) # Comment out this line and see what results it produces
        return o2
# Inputs to the model
x1 = torch.randn(1, 2)
