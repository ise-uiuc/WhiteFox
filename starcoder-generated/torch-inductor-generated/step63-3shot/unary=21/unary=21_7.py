
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(100, 100, 1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 100, 12)
