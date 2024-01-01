
class Model(torch.nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.conv = torch.nn.modules.conv.Conv2d(inp, outp, 3, stride=2, padding=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        out = self.conv(x)
        out = self.tanh(out)
        return out
# Inputs to the model
inp = 100
out = 100
