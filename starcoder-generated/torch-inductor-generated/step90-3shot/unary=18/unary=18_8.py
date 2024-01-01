
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input0 = torch.nn.Parameter(torch.randn(1, 3, 224, 224), requires_grad=False)
    def forward(self, x1):
        v1 = self.input0
        v2 = self.conv1(v1)
        v3 = self.conv2(v1) + v2
        return v1, v3
# Inputs to the model
x1 = torch.randn(3, 3, 224, 224)
