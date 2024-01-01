
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 24, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.elu = torch.nn.ELU(alpha=1.0)
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.elu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 127, 127)
