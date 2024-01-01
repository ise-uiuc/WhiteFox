
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(456, 784, bias=False)
        self.l2 = torch.nn.Linear(784, 1024, bias=False)
        self.l3 = torch.nn.Linear(1024, 192, bias=True)
    def forward(self, x):
        x1 = torch.reshape(x, (x.shape[0], -1))
        x2 = self.l1(x1)
        x3 = torch.reshape(x6, (x2.shape[0], -1))
        x4 = self.l2(x3)
        x5 = torch.reshape(x4, (x.shape[0], -1))
        x6 = self.l3(x5)
        return x6
# Inputs to the model
x = torch.randn(35, 16, 7, 3)
