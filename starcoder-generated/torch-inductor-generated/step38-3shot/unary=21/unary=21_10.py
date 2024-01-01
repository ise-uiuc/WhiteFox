
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, 3, padding=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = self.conv(t1)
        t3 = self.conv(t2)
        t4 = torch.tanh(t3)
        return
# Inputs to the model
x = torch.randn(2, 1, 512, 512)
