
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        a = torch.cat((input, input), dim=0)
        b = torch.cat([a, a, a], dim=1)
        c = torch.cat([b, b, b], dim=0)
        d = torch.cat([c, c], dim=1).view(b.size(0),-1).mean(dim=1).pow(2)
        return d
# Inputs to the model
input = torch.randn(1, 2)
