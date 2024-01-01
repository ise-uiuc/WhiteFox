
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        v = []
        for i in range(3):
            v.append(x[i, :, :])
            v.append(x[i+1, :, :])
        return torch.cat(v, 1)
# Inputs to the model
x = torch.randn(3, 3)
