
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, (5, 5))
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(64, 16, (5, 5))
        self.bn3 = torch.nn.BatchNorm2d(16)
        self.conv2 = torch.nn.Conv2d(32, 1, (7, 7))
        self.bn2 = torch.nn.BatchNorm2d(1)
    def forward(self, input_tensor):
        output = self.conv1(input_tensor)
        output = self.bn1(output)
        output = torch.nn.functional.relu(output)
        output = torch.nn.functional.relu(self.bn3(self.conv3(output)))
        output = torch.nn.functional.relu(self.bn2(self.conv2(output)))
        return output
# Inputs to the model
input_tensor = torch.randn(1, 1, 16, 16)
