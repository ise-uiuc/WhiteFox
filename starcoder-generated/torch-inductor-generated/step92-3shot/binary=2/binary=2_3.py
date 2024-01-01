
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(18, 3, 1, stride=1, padding=1)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = v1 * 0.3
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 3
        return v5
# Inputs to the model
input = torch.randn(1, 3, 30, 30)
