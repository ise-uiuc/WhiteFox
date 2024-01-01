
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 1)
        self.bn1 = torch.nn.BatchNorm2d(2)
        t = (3, 3, 3, 2)
        self.conv2 = torch.nn.ConvTranspose2d(2,2,2)
        self.bn2 = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.randn(t) # use torch.randn() to bypass error "could not broadcast input array from shape"
        x = self.conv2(x)
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
