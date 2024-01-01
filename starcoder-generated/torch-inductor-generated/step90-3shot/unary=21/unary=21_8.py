
class ModelTanhFlatten(torch.nn.Module):
    def __init__(self):
        super(ModelTanhFlatten, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=2)
    def forward(self, input):
        x = self.conv1(input)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.conv3(x)
        x = torch.tanh(x)
        x = x.flatten(1)
        x = torch.nn.Linear(512, 100)(x)
        x = torch.nn.Linear(100, 8)(x)
        x = torch.nn.Linear(8, 10)(x)
        return x
# Inputs to the model
input = torch.randn(1, 3, 32, 32)
