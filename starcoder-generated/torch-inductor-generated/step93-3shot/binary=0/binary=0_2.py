
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7)
    def forward(self, x1, dim=None, keepdim=True):
        v1 = self.conv(x1)
        if dim == None:
            dim = torch.randint(high=4, size=())
        v2 = torch.norm(v1, 2, dim, keepdim)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 5, 5)
