
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        v1 = torch.rand(3, 3)
        v2 = torch.mm(v1, v1) + v1
        return v2
