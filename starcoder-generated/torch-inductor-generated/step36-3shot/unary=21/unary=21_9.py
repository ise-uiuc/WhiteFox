
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv2d = nn.Conv2d(3, 32, 2)
        self.t5 = nn.Tensor = torch.empty([1,1,1,1])
    def forward(self, x):
        v1 = self.conv2d(x)
        v1 = self.t5.tanh(v1)
        return v1
# Inputs to the model
x = torch.randn(16, 3, 31, 31)
