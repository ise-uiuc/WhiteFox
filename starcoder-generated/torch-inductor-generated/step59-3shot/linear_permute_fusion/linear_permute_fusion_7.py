
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1)
    def forward(self, input):
        b_0, c_0, d1, d2 = input.shape
        out = self.conv(input)
        v6 = out.shape
        return out
# Inputs to the model
input = torch.randn(16, 2, 1, 4, device='cpu')
