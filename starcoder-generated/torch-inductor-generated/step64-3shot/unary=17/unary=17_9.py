
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(31, 5, (1, 1), stride=(1, 1), padding=(0, 0), output_padding=(0, 1), dilation=(1, 1), groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 31, 312, 312)
