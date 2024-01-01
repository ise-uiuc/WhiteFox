
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 513, 1, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        o1 = self.conv(x)
        q = torch.mean(torch.mean(o1))
        o2 = torch.tanh(q)
        o3 = torch.tanh(o2)
        o4 = torch.mean(o1)
        o5 = o3 + o4
        return o5
# Inputs to the model
x = torch.randn(1, 5, 31, 255)
