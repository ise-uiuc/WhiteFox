
class Model(torch.nn.Module):
    def __init__(self, in_channels=24, out_channels=24):
        super().__init__()
        self.pad1 = torch.nn.ReplicationPad2d((1,2,3,4))
        self.pool = torch.nn.MaxPool2d(2)
    def forward(self, x1, other=None):
        x2 = self.pad1(x1)
        v1 = self.pool(x2)
        if other is None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 24, 48, 48)
