
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=2)
        self.relu1 = torch.relu
        self.conv2 = torch.nn.Conv2d(32, 1, 1)
    def forward(self, x):
        return self.conv2(self.relu1(self.conv1(x))) - torch.randn(1)
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
