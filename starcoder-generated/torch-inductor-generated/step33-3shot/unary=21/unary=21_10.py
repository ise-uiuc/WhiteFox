
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1a_3x3 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.tanh_1 = torch.nn.Tanh()
    def forward(self, x0):
        v1 = self.conv2d_1a_3x3(x0)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x0 = torch.randn(1, 3, 128, 128)
