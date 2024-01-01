
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=0, stride=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, padding=0, stride=2, dilation=1)
        self.fc1 = torch.nn.Linear(8, 8)
        self.fc2 = torch.nn.Linear(8, 8)
    def forward(self, x):
        x = (self.conv1(x)).view(3, -1)
        x = (self.fc1(x)).view(x.size(0), -1)
        return torch.tanh(self.fc2(x))
# Inputs to the model
x = torch.randn(2, 3, 8, 8)
