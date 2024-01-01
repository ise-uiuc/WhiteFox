
class Model(torch.nn.Module):
    def __init__(self):
        # Nest two convolution modules, and connect out_channels of the first with in_channels of the second. 
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 2, 2)
        self.conv1 = torch.nn.Conv2d(2, 3, 3)
        self.conv2 = torch.nn.Conv2d(5, 8, 5)
    def forward(self, x): 
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 8, 8)
