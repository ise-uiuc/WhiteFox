
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, (1,1), stride=(1,1), padding=(0,0))
        self.conv2 = torch.nn.Conv2d(6, 6, (5,5), stride=(1,1), padding=(2,2))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = F.avg_pool2d(v2, 3, 3, 0)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
