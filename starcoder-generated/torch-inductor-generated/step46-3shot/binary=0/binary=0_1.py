
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
        self.fc = torch.nn.Linear(8, 1)
    def conv_block(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.fc(v2)
        return v3
    def forward(self, x1, x2):
        # Add feature after conv2 layer in the forwarding path.
        v1 = self.conv_block(x1)
        v2 = self.conv_block(x2)
        v3 = v1 * v2
        v4 = v3 + v2
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
