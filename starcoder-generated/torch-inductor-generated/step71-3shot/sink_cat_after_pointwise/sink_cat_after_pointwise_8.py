
class ModuleForTest(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 2, stride=2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(16*8*8, 32)
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
x = torch.randn(1, 3, 224, 61, requires_grad=True)
