
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        s = torch.nn.functional.conv2d(x1, torch.rand([3, 4, 3, 3], device='cpu'))
        t = torch.nn.functional.batch_norm(s)
        return t
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
