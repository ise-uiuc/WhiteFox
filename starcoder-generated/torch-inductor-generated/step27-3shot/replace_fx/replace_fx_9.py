
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.tensor([float('Inf')])
        self.t2 = torch.tensor([float('Nan')])
        self.x2 = torch.tensor([float('Inf')])
        self.x3 = torch.tensor([float('NaN')])
        self.x4 = torch.tensor([-float('Inf')])
        self.x5 = torch.tensor([-float('NaN')])
        #self.rand = torch.empty(2, dtype=torch.float64).uniform_(0, 1.0).squeeze()
    def forward(self, x1):
        x2 = x1 + self.x2
        x3 = x1 * self.x3
        x4 = x1 ** (self.x4)
        x5 = x1 ** (self.x5)
        x6 = x1 / self.x3
        return torch.tensor([1.0])
# Inputs to the model
x1 = torch.randn(2)
