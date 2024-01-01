
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.tanh(x1)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x3 = self.sigmoid(x2)
        return x3
# Inputs to the model
x = torch.randn(1, 128, 300, 300)
