
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t = torch.mm(input, input)
        return torch.cat([t, t])
# Inputs to the model
input = torch.randn(3, 4)
