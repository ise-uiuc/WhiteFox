
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 2, bias=False)
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, input_tensor):
        return torch.cat((self.conv(input_tensor), self.bn(input_tensor)), dim=1)
# Inputs to the model
x = torch.randn(1, 2, 5, 5)
