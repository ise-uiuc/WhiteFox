
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24,24,kernel_size=(1, 9),stride=(1, 2),bias=False,groups = 24)
        self.conv2 = torch.nn.Conv2d(24,24,kernel_size=(5, 1),stride=(1, 2),bias=False,groups = 24)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 24, 64, 64)
