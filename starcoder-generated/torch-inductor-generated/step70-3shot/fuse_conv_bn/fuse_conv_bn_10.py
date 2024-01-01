
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3,4,3)
        self.conv2 = torch.nn.Conv2d(4,5,3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
