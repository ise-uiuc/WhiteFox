
class Model(torch.nn.Module):
    # The stride and zero-padding are deliberately set to some different values here to ensure there will be differences in the generated model that produce different input tensors
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=77)
        self.bn = torch.nn.BatchNorm1d(4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2.add(3)
        v4 = v3.clamp(0, 6)
        v5 = v4 - 3
        v6 = v5 * 3
        v7 = v6 / 9.0
        return v7
# Inputs to the model
x1 = torch.randn(128, 3, 123)
