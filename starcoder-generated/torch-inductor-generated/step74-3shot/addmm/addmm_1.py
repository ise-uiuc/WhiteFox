
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.randn(3, 3, requires_grad=True)
        self.inp = torch.randn(3, 3)
    def forward(self):
        return torch.mm(self.x, self.inp)
        return F.leaky_relu(x) + self.x + self.inp
class NewRelu(torch.nn.Module):
    def forward(self, x):
        return F.relu(x)
