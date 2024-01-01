
class A(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, kernel_size=5)
    def forward(self, input):
        y = self.conv(input)
        t = input.permute(1, 0, 2, 3)
        z = t.reshape((1, 20))
        x = y + z
        out = torch.rand_like(x)
        return out
# Inputs to the model
input = torch.randn(1, 3, 256, 256)
