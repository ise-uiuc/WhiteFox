

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(4, 3, 3, bias=False)
    def forward(self, x1):
        s = self.conv(x1)
        t = torch.nn.functional.batch_norm(s, 5)
        return t
# Inputs to the model
x1 = torch.randn(1, 4, 1, 4, 4)
