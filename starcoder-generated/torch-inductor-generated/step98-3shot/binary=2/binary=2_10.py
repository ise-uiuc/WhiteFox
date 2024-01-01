
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, (3, 3), stride=(2,1), padding=(1,2))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
        return v2
# Inputs to the model
x = torch.randn(1, 8, 102, 102)
