
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose1d(1, 1, 10)
    def forward(self, x1):
        t1 = torch.nn.functional.dropout(x1, p=0.5)
        x2 = torch.rand_like(t1)
        x3 = self.conv(x1)
        x4 = self.conv(x1)
        x5 = self.conv(t1)
        x6 = self.conv(t1)
        return x7
# Inputs to the model
x1 = torch.randn(1, 1, 10)
