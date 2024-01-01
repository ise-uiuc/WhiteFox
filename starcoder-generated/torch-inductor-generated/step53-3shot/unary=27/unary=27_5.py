
class Model1(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 19, kernel_size=(3,), stride=(1,), padding=(1,))
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, self.min, self.max)
        return v2
min = None
max = None
# Inputs to the model
x1 = torch.randn(2, 3, 64)
