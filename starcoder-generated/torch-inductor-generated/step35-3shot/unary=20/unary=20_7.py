
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.Conv2d(2, 1, 1), torch.nn.Sigmoid())
        self.deconv2 = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.Conv2d(2, 1, 1), torch.nn.Sigmoid())
    def forward(self, x):
        return self.deconv(x) + self.deconv2(x)
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
