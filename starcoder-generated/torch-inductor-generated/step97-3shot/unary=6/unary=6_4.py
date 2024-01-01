
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.conv_b = torch.nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.conv_c = torch.nn.Conv2d(64, 128, 5, stride=1, padding=2)
    def forward(self, x1):
        t1 = self.conv_a(x1)
        t2 = self.conv_b(t1)
        t3 = self.conv_c(t2)
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
