
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 1, stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1.permute(1,0,2,3)
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
