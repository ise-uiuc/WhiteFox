
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, input1=False, input2=False, input3=False):
        x1 = torch.randn(1, 1, 64, 64)
        x2 = torch.randn(1, 1, 64, 64)
        x3 = torch.randn(1, 1, 64, 64)
        if input1 == None:
            input1 = x1
        if input2 == None:
            input2 = x2
        if input3 == None:
            input3 = x3
        v1 = self.conv(input1)
        v2 = v1 + input2
        v3 = v2 + input3
        return v3
