
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1, 64, 3, bias=False, stride=2, padding=1)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        t1 = self.conv_t(x)
        t2 = torch.sum(t1)
        t3 = self.softmax(t1)
        return t3
# Inputs to the model
x = torch.randn(1, 1, 52, 95, 120)
