
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input, dim0, dim1):
        output = torch.cat([input[..., -dim0:], input, input[..., :dim1]], dim = -1)
        return output
# Inputs to the model
input = torch.randn(1, 3, 96, 96)
dim0 = 3
dim1 = 3
