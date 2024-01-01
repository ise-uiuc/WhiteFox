
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=50, out_channels=3, kernel_size=9, stride=1, padding=4)
    def forward(self, x1):
        x0 = torch.zeros(1, 3, 1304, 1008).to("cpu")
        v1 = self.conv(x0)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 50, 1008, 1304)
