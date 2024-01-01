
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.rand(1, 1)
    def forward(self, x2):
        v1 = torch.cat([torch.mm(self.x1, x2), torch.mm(self.x1, x2)], 1)
        return torch.cat([torch.mm(self.x1, x2), v1], 1)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(4, 1)
