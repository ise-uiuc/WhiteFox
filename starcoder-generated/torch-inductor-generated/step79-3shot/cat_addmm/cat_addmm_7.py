
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.conv = nn.Conv2d(4, 4, 4)
    def forward(self, x):
        x = self.linear(x)
        y = self.conv(x)
        s = torch.jit.script(y)
        w = torch.conv2d(y, s)
        h = torch.conv2d(w, s, stride=1)
        q = torch.conv2d(y, s, padding=1)
        return h
# Inputs to the model
x = torch.randn(1, 1, 1, 2)
