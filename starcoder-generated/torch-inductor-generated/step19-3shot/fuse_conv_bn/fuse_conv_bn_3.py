
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x2):
        s2 = self.conv(x2)
        t2 = torch.nn.functional.tanh(s2)
        u2 = torch.nn.functional.tanh(s2)
        t2.retain_grad()
        u2.retain_grad()
        v2 = t2.view_as(u2)
        y2 = (s2 + u2 + v2)
        return y2
# Inputs to the model
x2 = torch.randn(5, 5, 1, 1)
