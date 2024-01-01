
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out=[]
        for i in range(3):
            out.append(torch.cat([x, x], dim=1))
        out = torch.cat(out, dim=1)
        out = out.reshape(x.shape[0], x.shape[1], -1)
        out = torch.relu(out)
        z = torch.cat([out, out, out], dim=1)
        z = torch.tanh(z)
        return z
x = torch.randn(1, 2, 3)
