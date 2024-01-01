
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(28, 40)
        self.avg = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = torch.nn.Flatten()
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.avg(v1)
        v3 = self.flatten(v2)
        v4 = self.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
