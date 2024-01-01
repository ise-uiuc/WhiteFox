
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = torch.nn.Identity()
    def forward(self, x1):
        v1 = x1
        v1 = self.identity(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 3, 4)
