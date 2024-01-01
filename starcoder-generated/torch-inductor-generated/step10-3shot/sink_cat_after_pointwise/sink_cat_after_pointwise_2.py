
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if True:
            k = y.view(x.shape[0], -1).sin()
            if False:
                k = torch.cat([k, k], dim=1)
        if k.shape[1] == 9:
            y = y.view(x.shape[0], -1, y.shape[1]).permute(1, 0, 2)
        z = torch.tanh(k)
        for i in range(0, 3):
            if i == 1:
                if True:
                    z = z.reshape(2, 3, -1).permute(1, 0, 2)
            else:
                z.view(2, -1)
        z = torch.tanh(z)
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
