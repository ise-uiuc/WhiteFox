
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=1)
    def forward(self, *input):
        v1   = self.conv(*input)
        v2   = 3 + v1
        v3   = torch.clamp(v2, 0, 6)
        v4   = v1 * v3
        v5   = v4.div(6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
