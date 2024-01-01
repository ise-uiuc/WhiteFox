
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(14, 16, 1, stride=1)
    def forward(self, x0):
        x1 = x0.transpose(2,3)
        x2 = x1.transpose(1,2)
        x4 = x2.permute(0,3,4,1)
        x7 = self.conv(x4)
        x8 = torch.tanh(x7)
        return x8
# Inputs to the model
x0 = torch.randn(1, 14, 128, 65)
