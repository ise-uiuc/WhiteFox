
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,64,1,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8,out_channels=128,kernel_size=7,stride=1,padding=3)
    def forward(self, x1):
        v0 = x1
        v1 = self.conv1(v0)
        a1 = torch.relu(v1)
        v2 = self.conv2(a1)
        a2 =  torch.tanh(v2)
        return a2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
