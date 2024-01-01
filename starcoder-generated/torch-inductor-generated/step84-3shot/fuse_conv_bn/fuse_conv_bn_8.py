
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.bias = torch.nn.Parameter(torch.randn(1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + self.bias
        x = x.relu() + self.bias # Add one more layer to trigger the pattern
        return x
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
