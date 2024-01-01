
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        for loopVar1 in range(5):
            x1 = torch.nn.functional.batch_norm(x1)
        v1 = x1
        for loopVar1 in range(7):
            v1 = torch.nn.functional.batch_norm(v1)
        return torch.cat([v1, v1, v1], 1)
