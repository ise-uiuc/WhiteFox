
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 256, (3, 3), padding=1, padding_mode='replicate')
        self.conv2 = torch.nn.Conv2d(1, 256, (1, 1), padding=0, dilation=2, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1 + v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 12, 12)
