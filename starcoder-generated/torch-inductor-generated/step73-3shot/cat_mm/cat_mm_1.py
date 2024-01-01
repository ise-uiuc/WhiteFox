
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input2):
        v1 = torch.cat([torch.mm(input2, input2), torch.mm(input2, input2)], 1)
        v2 = torch.cat([torch.mm(input2, input2), torch.mm(input2, input2)], 1)
        v3 = torch.cat([torch.mm(input2, input2), torch.mm(input2, input2)], 1)
        v = []
        v.append(v1)
        v.append(v2)
        v.append(v3)
        v4 = torch.mm(input2, input2)
        v5 = torch.mm(input2, input2)
        v6 = torch.mm(input2, input2)
        return torch.cat([v1, v2, v3, v4, v5, v6], 1)
# Inputs to the model
input2 = torch.randn(2, 2)
