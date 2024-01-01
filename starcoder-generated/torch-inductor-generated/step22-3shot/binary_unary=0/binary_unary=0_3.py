
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=16, padding=15)
    def forward(self, input):
        v1 = torch.relu(self.conv1(input))
        v2 = torch.tanh(v1)
        v3 = torch.matmul(v2, v2)
        return v3

# Inputs to the model
x1 = torch.randn(1, 8, 28, 28)

