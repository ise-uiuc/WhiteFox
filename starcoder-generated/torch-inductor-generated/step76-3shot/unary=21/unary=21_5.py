
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.linear1 = torch.nn.Linear(9216, 2048)
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = x.reshape(-1, 9216)
        x = torch.tanh(self.linear1(x))
        return x
# Inputs to the model
x = torch.randn(128, 1, 224, 224)
