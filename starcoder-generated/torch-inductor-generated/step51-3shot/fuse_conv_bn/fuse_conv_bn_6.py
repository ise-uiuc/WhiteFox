
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(2, 2, 2)
        self.bn = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU(inplace=False)
        self.prelu = torch.nn.PReLU(num_parameters=2, init=0.1)
    def forward(self, x):
        x = self.relu(self.bn(self.c(x)))
        x = self.relu(self.prelu(self.c(x)))
        return x
# Inputs to the model
x = torch.randn(1, 2, 8, 8)
