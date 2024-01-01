
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        v1 = torch.mm(input, input)
        v2 = torch.mm(v1, v1)
        v3 = torch.mm(v2, v2)
        v4 = torch.mm(v3, v3)
        v5 = torch.mm(v4, v4)
        v6 = torch.mm(v5, v5)
        return torch.cat([v1, v2, v3, v4, v5, v6], 1)
# Inputs to the model
input = torch.randn(2, 2)
