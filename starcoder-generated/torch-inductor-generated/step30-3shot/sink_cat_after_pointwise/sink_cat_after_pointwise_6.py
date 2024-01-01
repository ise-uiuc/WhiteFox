
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = linear = nn.Linear(2, 3)
        self.conv = nn.Conv2d(3, 8, 5)

    def forward(self, x):
        out = x.relu()
        out = self.linear(out)
        tmp = out[0] # get the first element from the tensor output of self.linear
        out = tmp.clamp(min=0) # clamp the tensor into range [0, inf)
        out = self.conv(out)
        out = out.relu()
        out = out[:, out.shape[1] // 2:]
        out = out.sum((-1, -2))
        return out
# Inputs to the model
x = torch.randn(1, 2, 10, 10)
