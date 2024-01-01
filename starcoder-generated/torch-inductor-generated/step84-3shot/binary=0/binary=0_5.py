
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(35, 43, 2, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(43, 51, 2, stride=1, padding=1)
    def forward(self, x2: torch.Tensor, padding2 = None):
        v1 = self.conv1(x2)
        if padding2 == None:
            padding2 = torch.nn.ReplicationPad2d(1)
        v2 = padding2(v1)
        v3 = self.conv2(v2)
        v4 = v3 + torch.randn(v3.shape)
        return v4
# Inputs to the model
x2 = torch.randn(1, 35, 64, 64)
