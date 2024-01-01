
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(32, 7))
    def forward(self, v1):
        split = torch.split(v1, [1, 1, 1], dim=1)
        if split[0].shape[-1] > 2:
            temp = split[0]
        elif split[2].shape[-1] > 2:
            temp = split[2]
        else:
            temp = split[1]
        return torch.cat([temp, v1], 1)
# Inputs to the model
x1 = torch.randn(1, 8, 32)
