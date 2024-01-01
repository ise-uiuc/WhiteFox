
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        for i in range(8):
            # TODO: Concatenating a tensor to itself
            # v2 = torch.cat([v2, v2], 1)
        return torch.cat([v1, v2, v1, v2, v1, v2, v1], 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
