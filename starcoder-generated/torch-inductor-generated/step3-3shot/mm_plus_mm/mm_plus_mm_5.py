
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = torch.nn.Conv2d(100,284,16,stride=16)
    def forward(self, input1):
        t1 = self.conv2d(input1)
        return t1
# Inputs to the model
input1 = torch.randn(1, 100, 7,7)
