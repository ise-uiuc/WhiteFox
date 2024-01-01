
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.pad(x1, (1, 2, 3, 4), "constant", value=1)
        v2 = v1.permute(0, 2, 1, 3)
        output = v1 - v2
        return output.permute(0, 2, 1, 3)
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
