
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=2, padding=0, groups=4)
        self.conv2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)   
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 16, 4, 4)
