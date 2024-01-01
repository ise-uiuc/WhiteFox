
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=6, bias=False)
        self.linear = torch.nn.Linear(3072, 512)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = F.relu(v1)
        v1 = self.linear(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
