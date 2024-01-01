
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 3, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(7, 3, 3, stride=2, padding=2)
    def forward(self,x1,x2=torch.randn(1, 7, 1024, 1024),padding1=-20.7):
        v1=self.conv(x1)
        if x2 == None:
            x2 = torch.randn(v1.shape)
        v2=self.conv2(x2)
        v3=v1+v2
        v4=v3.permute(0, 1, 3, 2).contiguous().squeeze(2)
        return v4
# Inputs to the model
x1=torch.randn(1, 8, 128, 128)
