
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(1, 2)
    def forward(self, x1, x2):
        v = torch.cat((torch.mm(x1, self.weight), torch.mm(x2, self.weight)), 1)
        return v
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
