
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.tensor(1.)
    def forward(self, x):
        dim2 = x.shape[1]
        y = torch.cat((torch.zeros(x.shape[0], 2, x.shape[2], device=self.t1.device), x, x), dim=1)
        y = y * self.t1.view(1, dim2, 1)
        y = y.view(y.shape[0], -1)
        return y.relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
