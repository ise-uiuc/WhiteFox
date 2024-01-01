
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(16, 8, kernel_size=1, stride=1, bias=False)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = v1 + v1
        v3 = torch.nn.functional.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 224, 224)
