
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = torch.nn.Conv2d(1, 16, 3, stride=(4, 4), padding='same')
        self.conv_b = torch.nn.Conv2d(1, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_a(x1)
        v2 = self.conv_b(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
