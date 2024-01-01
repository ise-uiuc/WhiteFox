
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        a1 = torch.cat([t1, t1], 1)
        a2 = torch.cat([a1, a1], 1)
        a3 = torch.cat([a2, a2], 1)
        a4 = torch.cat([a3, a3], 1)
        return torch.cat([a4, a4], 1)
# Inputs to the model
input1 = torch.randn(4, 2)
input2 = torch.randn(2, 128)
