
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 25, 3, stride=1, padding =(1,2,2,1))
        #self.conv2 = torch.nn.Conv2d(25, 50, 3, stride=2, padding=(0,3,1,1))
        #self.conv3 = torch.nn.Conv2d(5, 10, 3, stride=3, padding=(1,3,2,1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        #v2 = self.conv2(v1)
        #v3 = self.conv3(x1)
        v4 = torch.sub(v1, 4)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64, 64)
