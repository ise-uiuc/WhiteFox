
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 2, 1, stride=1, padding=1)
    def forward(self, x1, other=None, bias=None):
        s1 = self.conv(x1)
        out = s1 + other
        out += 1
        if not bias is None:
            v1 = out.transpose(-1, 0)
            v1 += bias
            v2 = v1 * other
            out = v2 + bias
        return out
# Inputs to the model
x1 = torch.randn(3, 8, 4)
x2 = torch.zeros(2, 3, 8)
