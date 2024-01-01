
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.randn(3, 3)
        self.x2 = torch.randn(3, 3)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2) \
            + torch.mm(self.x1, self.x2)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
