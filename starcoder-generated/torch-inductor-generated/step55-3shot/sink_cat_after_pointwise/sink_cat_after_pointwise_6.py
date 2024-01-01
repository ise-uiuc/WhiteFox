
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.b = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.c = torch.nn.Conv2d(3, 64, 3, stride=1, padding=(1, 1), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.d = torch.nn.Conv2d(3, 64, 3, stride=1, padding=(0, 0, 0, 0), dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        y = torch.relu(self.a(x))
        y = torch.relu(self.b(y))
        y = torch.relu(self.c(y))
        y = torch.relu(self.d(y))
        return y
# Inputs to the model
x = torch.randn(2, 3, 10, 12)
