
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 3, 1, stride=1, padding=0)
        self.relu = torch.nn.ReLU()
    def forward(self, x1, other=1, padding1=None, padding2=None, padding3=None, padding4=None, padding5=None, padding6=None, padding7=None, padding8=None, padding9=None, padding10=None, padding11=None, padding12=None, padding13=None, padding14=None, padding15=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = self.relu(v1)
        if padding2 == None:
            padding2 = torch.randn(v2.shape)
        return v2
# Inputs to the model
x1 = torch.randn(1, 7, 28, 28)
