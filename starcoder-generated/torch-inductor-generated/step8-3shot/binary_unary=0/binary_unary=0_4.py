
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = torch.mul(x, x)
        v2 = torch.mul(v1, v1)
        v3 = 1 + v1
        v4 = torch.mul(v1, v3)
        return v2
x = 0
