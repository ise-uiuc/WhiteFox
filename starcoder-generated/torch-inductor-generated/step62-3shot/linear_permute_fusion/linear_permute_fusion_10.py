
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Conv2d(1, 2, 3, groups=1)
    def forward(self, x1):
        x1 = F.relu(x1)
        x1 = x1.permute(0, 2, 3, 1).contiguous()
        x1 = F.relu(self.linear(x1))
        v2 = np.array([3, 1, 0, 2])
        v2 = v2 + 0
        v2 = x1.permute(0, 2, 1, 3)
        return x1[v2]
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
