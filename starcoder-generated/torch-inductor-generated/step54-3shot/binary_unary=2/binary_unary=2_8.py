
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(33, 8, 2, stride=1, padding=2)
    def forward(self, input1):
        v1 = self.conv1(input1)
        v2 = v1 - -28.15
        v3 = F.relu(v2)
        return v3
# Inputs to the model
input1 = torch.randn(1, 33, 55, 11)
