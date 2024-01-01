
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        v1 = torch.mm(input1, input2)
        v2 = torch.mm(input3, input4)
        v3 = v1 + v2
        output = v1 * v2 * v3
        return output
# Inputs to the model
input1 = torch.randn(75, 75)
input2 = torch.randn(75, 75)
input3 = torch.randn(75, 75)
input4 = torch.randn(75, 75)
