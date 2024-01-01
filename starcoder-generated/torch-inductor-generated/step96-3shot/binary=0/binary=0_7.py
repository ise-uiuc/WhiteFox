
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv01 = torch.nn.Conv2d(9, 6, 3, stride=1, padding=1)
        self.conv02 = torch.nn.Conv2d(10, 8, 3, stride=1, padding=1)
    def forward(self, x1, x2, other=1, other1=1, other2=1, padding1=None, padding2=None):
        v1 = self.conv01(x2) 
        v2 = self.conv02(x1) # Conv01's output is an input to Conv02
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        v3 = v1 + v2
        return v3
# Input to the model
x1 = torch.randn(1, 10, 16, 16)
x2 = torch.randn(1, 9, 64, 64)
