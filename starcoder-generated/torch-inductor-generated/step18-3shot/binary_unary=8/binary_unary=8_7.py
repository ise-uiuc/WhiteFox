
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=3, padding=1)
    def forward(self, x1):
        # Pad the input tensor to allow for non-overlapping outputs
        t1 = torch.nn.functional.pad(x1, (1, 1, 1, 1), "constant", 0)
        v1 = self.conv(t1)
        v2 = self.conv(t1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
