
# Definition of a module with 2 pointwise convolution layers, one following the other
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,32,3,1,'same')
        self.conv2 = torch.nn.Conv2d(32,1,3,1,'same')
    def forward(self,x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
