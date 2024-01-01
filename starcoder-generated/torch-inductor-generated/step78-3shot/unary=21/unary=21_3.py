
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3,1,1)
        self.conv2d2 = torch.nn.Conv2d(3,1,1)
    def forward(self, x):
        v1 = torch.tanh(self.conv2d1(x))
        v2 = self.conv2d2(v1)
        return v2
# Inputs to the model
x = torch.rand(1,3,224,224)
