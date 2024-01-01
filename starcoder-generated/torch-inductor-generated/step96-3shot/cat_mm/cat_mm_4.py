
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3):
        out = torch.cat([torch.mm(input1, torch.transpose(input2, 1, 0)), torch.mm(input1, input2), torch.mm(input1, torch.transpose(input2, 1, 0))], 1)
        out = torch.transpose(out, 0, 1)
        return out
# Inputs to the model
input2 = torch.randn(2, 5)
input1 = torch.randn(2, 3)
input3 = torch.randn(2, 4)
