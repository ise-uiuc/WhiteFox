
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(4, 5, kernel_size= 2, stride=2,padding=1, bias=True)
        self.convtranspose = torch.nn.ConvTranspose2d(5, 1, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1.repeat(1,1,2,2)
        v3 = self.convtranspose(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
# Model end


# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtranspose = torch.nn.ConvTranspose2d(6, 1, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.convtranspose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to model
x1 = torch.randn(1, 6, 4, 4)
