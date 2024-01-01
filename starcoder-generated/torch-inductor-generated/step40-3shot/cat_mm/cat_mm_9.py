
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        out = []
        for input1 in input1:
            out.append(input2)
        return out
# Inputs to the model
input1 = torch.randn((2, 2))
input2 = torch.randn(2, 2)
