
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 512, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        print('v1.grad: ', v1.requires_grad)
        v2 = self.tanh(v1)
        v3 = torch.tanh(v1)
        return v3
# Inputs to the model
x = torch.randn(1, 128, 64, 64)
