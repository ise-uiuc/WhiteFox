
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.t1 = torch.mean(torch.rand(2,2))
        self.t2 = torch.mean(torch.rand(2,2))
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x4)
        v2 = torch.mm(x3, x2)
        v3 = 0.1 * v1 + 0.05 * v2 + self.t1 + self.t2
        return v3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
