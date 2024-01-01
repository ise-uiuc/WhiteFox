
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = torch.mm
    def forward(self, x1, x2):
        v1 = self.mm(x1, x2)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
