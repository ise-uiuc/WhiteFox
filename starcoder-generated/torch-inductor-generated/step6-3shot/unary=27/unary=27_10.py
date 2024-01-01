
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=8, out_channels=3, kernel_size=3, stride=2, padding=1)
    def forward(self, input):
        x = self.conv(input)
        min = 0.6
        max = 0.6
        y_min = torch.clamp_min(x, min=min)
        y_max = torch.clamp_max(x, max=max)
        return y_max
# Inputs to the model
input = torch.randn(3, 8, 64)
