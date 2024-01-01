
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
    def forward(self, input0=None, input1=None, input2=None, input3=None):
        v1 = self.conv(input0)
        v2 = torch.randn(v1.shape)
        v3 = v1 + v2
        if input1 is None:
            input1 = torch.randn(v3.shape)
        if input2 is None:
            input2 = torch.randn(v3.shape)
        v4 = v3 + input1
        v5 = v4 + input2
        if input3 is None:
            input3 = torch.randn(v5.shape)
        v6 = v5 + input3
        return v6
