
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
 
    def forward(self, x):
        v1 = self.pool(self.conv2(x))
        v2 = self.pool(self.conv1(x))
        v3 = torch.mm(v1, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
kwargs = {'inp' : torch.randn(8, 1)}
