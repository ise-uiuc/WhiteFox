
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(768, 768, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(768, 768, (1, 1), stride=(1, 1), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v5 = v2 - 0
        return v5
# Inputs to the model
x1 = torch.randn(1, 768, 8, 8)
