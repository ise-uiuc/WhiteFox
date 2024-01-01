
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.randn(10)
        self.x1 = torch.randn(10)
    def forward(self, x2):
        y1 = torch.mm(x2, x2) + self.x1 + x2 + self.v1
        return x2 + y1
# Inputs to the model
x2 = torch.randn(10, 10)
