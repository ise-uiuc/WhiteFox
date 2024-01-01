
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.conv_first = torch.nn.Conv2d(1,16,5)
        self.conv4 = torch.nn.Conv2d(16, 16, 5)
        torch.manual_seed(2)
        self.bn = torch.nn.BatchNorm2d(16)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv_first(x)
        x = self.conv4(x)
        x = self.bn(x)
        return self.activation(x)
# Inputs to the model
x = torch.randn(1, 1, 12, 12)
