
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.r2 = torch.nn.ReLU()
        self.r1 = torch.nn.ReLU()
    def forward(self, x1, x2):
        return self.r2(self.r1(torch.sum(x1 + x2)))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
