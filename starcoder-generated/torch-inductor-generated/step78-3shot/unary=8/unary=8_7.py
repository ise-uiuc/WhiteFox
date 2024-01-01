
# Note here we explicitly specify output padding as 1, so the output size should be (N, C, H, W) instead of (N, H, W, C)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(25, 19, 5, stride=1, padding=0, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1d(x1)
        v2 = v1 / 6
        return v6
# Inputs to the model
x1 = torch.randn(6, 12, 24)
