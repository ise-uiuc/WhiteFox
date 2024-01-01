
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = v1 - -1.1
        v3 = F.tanh(v2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
