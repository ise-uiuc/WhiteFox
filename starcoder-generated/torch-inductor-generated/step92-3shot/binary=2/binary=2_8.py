
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__() #
        self.conv2 = torch.nn.Conv2d(3, 1, 7, stride=1, padding=3)
    def forward(self, x): #
        v1 = self.conv2(x) #
        v2 = 3.14159
        v3 = v1 - v2
        return torch.relu(v3)
# Inputs to the model
x = torch.randn(1, 3, 22, 22)
