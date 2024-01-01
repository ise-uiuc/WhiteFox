
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 64, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 16, 5, stride=2, padding=2)
        self.conv5 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(8, 4, 3, stride=1, padding=1)

        # Insert your code here
        # Hint: use method.register_forward_hook() for self.register_forward_hook(function)

        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
