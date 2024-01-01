
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=(1, 1), stride=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
        self.conv4 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
        self.conv5 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
        self.conv6 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
        self.conv7 = torch.nn.Conv2d(32, 32, kernel_size=(1, 1), stride=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v0 = v0.add_((torch.relu(self.conv4(v3)).mul_((torch.relu(self.conv5(v3))))))
        v1 = torch.relu(self.conv6(v0))
        v2 = torch.relu(self.conv7(v1))
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
