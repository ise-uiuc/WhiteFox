
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        v1 = torch.mm(inp, inp)
        v2 = v1 + torch.eye(2, dtype=torch.float)
        return v2.size(0)
# Inputs to the model
inp = torch.eye(5, dtype=torch.float)
