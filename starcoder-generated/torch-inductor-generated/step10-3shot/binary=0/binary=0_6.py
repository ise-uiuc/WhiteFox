
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 4, 1, stride=1, padding=1)
    def forward(self, x, padding=None):
        v = self.conv(x)
        if padding == None:
            padding = torch.ones(v.shape)
        padding_ = torch.nn.functional.pad(padding, (1, 1, 1, 1))
        c = v * padding_
        return c
# Inputs to the model
x = torch.rand(8, 4, 64, 64)
