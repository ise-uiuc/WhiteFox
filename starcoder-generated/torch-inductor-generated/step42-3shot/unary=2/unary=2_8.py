
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.transpose(x1, 0, 1)
        v2 = torch.transpose(v1, 1, 2)
        v3 = torch.transpose(v2, 0, 1)
        v4 = torch.transpose(v3, 2, 3)
        v5 = torch.reshape(v4, (4, 384))
        v6 = torch.reshape(v5, (4, 27, 4, 16))
        return v6
# Inputs to the model
x1 = torch.reshape(torch.arange(4 * 27 * 4 * 16, dtype=torch.float32), (4, 1, 27, 4, 16))
