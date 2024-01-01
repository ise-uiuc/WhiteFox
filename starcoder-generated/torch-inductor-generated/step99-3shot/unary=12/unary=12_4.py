
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
        conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Sequential(
            conv1,
            torch.nn.Sigmoid(),
            conv2,
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(4, 1, 1, stride=1, padding=1),
            torch.nn.Sigmoid()
        )
    def forward(self, x1):
        v1 = self.conv3(x1)
        v2 = v1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
