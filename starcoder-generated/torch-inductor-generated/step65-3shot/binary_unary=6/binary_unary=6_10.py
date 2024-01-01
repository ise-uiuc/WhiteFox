
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.randn((3, 4)))
        v2 = v1 - 1
        v3 = torch.relu(v2)
        return v3