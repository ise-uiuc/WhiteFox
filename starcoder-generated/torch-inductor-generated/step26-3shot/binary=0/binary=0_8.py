
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, **kwargs):
        v1 = self.conv(x1)
        other1 = kwargs.get("other1", None)
        if other1 == None:
            other1 = torch.ones(v1.shape)
        other2= kwargs.get("other2", None)
        if other2 == None:
            other2 = torch.randn(v1.shape)
        v2 = v1 + other1 + other2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
