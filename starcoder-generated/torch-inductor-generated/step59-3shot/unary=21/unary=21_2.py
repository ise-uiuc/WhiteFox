
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pad1 = torch.nn.ReflectionPad2d(2)
        self.pad2 = torch.nn.ReflectionPad2d(2)
        self.conv1= torch.nn.Conv2d(2, 3, 5, stride=2, padding=0)
    def forward(self, x):
        v1 = self.pad1(x)
        v2 = self.pad2(v1)
        v3 = self.conv1(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 2, 26, 26)
