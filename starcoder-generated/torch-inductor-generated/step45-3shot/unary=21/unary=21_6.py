
class PatternModule(torch.nn.Module):
    def __init__(self, kernel_size):
        super(PatternModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.tanh(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
