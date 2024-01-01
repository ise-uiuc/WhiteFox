
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 32, 3)
    def forward(self, x1, x2):
        t1 = self.conv(x1)
        t2 = torch.exp(t1)
        return t2.mm(x2)
# Inputs to the model
x1 = torch.randn(50, 50, 1, 2000, device='cuda:0')
x2 = torch.randn(2000, 100, device='cuda:0')
