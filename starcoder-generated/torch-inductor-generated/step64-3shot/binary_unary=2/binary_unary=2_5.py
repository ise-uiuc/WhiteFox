
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(5, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = torch.nn.functional.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + 5.1
        v6 = torch.tanh(v5)
        return v6 
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
