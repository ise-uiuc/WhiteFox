
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv1 = torch.nn.Conv2d(16, 16, (1, 7), stride=(1, 1), padding=(0, 3))
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1):
        t1 = self.conv1(x1)
        y2 = self.conv2(t1)
        y1 = torch.tanh(y2)
        return y1
# Inputs to the model
x1 = torch.randn(1, 16, 24, 32)
