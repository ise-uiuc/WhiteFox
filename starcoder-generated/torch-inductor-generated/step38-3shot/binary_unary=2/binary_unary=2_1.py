
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3,128,kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv2 = torch.nn.Conv1d(128,512,kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.linear = torch.nn.Linear(11*11*512, 1000)
    def forward(self, x1, x2):
        x = x1 - x2
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.linear(v2.view(v2.size(0), -1))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 112, 112)
x2 = torch.randn(1, 3, 112, 112)
