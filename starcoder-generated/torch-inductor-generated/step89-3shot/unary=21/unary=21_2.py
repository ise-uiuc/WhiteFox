
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # comment
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
        self.tanh = torch.nn.Tanh()
        self.tanh2 = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = self.tanh(v2)
        v4 = self.tanh2(v3)
        return v4
# Inputs to the model
x = torch.randn(64, 3, 32, 32)
