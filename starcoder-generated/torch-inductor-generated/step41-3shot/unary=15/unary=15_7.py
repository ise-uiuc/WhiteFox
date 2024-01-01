
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 256, 1, stride=1, padding=1)
        self.relu3 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.relu(v1)
        v1 = self.conv2(v1)
        v1 = self.relu3(v1)
        v1 = v1.permute(0, 2, 3, 1)
        v1 = v1.flatten(start_dim=0, end_dim=1)
        v1 = torch.nn.functional.normalize(v1, p=2, dim=1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
