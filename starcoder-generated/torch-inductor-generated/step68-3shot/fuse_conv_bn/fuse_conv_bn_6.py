
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv_relu = torch.nn.Sequential(
        torch.nn.Conv2d(8, 8, 2),
        torch.nn.ReLU(),
        )
        torch.manual_seed(1)
        self.bn = torch.nn.BatchNorm2d(8) 
    def forward(self, x):
        x = self.conv_relu(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 8, 8, 8)
