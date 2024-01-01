
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        v = []
        v.append(torch.mm(input1, input2))
        v.append(torch.mm(input1, input2))
        v.append(torch.squeeze(torch.t(torch.squeeze(torch.t(input1)), 1), 2))
        return torch.cat(v, 2)
# Inputs to the model
input1 = torch.randn(2, 2)
input2 = torch.randn(2, 1)
