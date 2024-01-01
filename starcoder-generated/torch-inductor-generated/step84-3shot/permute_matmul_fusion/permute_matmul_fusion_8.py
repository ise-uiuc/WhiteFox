
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r = torch.nn.ReLU()
    def forward(self, x1, x2):
        v3 = x1.permute(0, 1, 3)
        v4 = torch.bmm(torch.bmm(x1, v3), x2.permute(0, 1, 3))
        v5 = x2.permute(0, 1, 3)
        return self.r(torch.bmm(v4, v5))
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 2, 2, 2)
