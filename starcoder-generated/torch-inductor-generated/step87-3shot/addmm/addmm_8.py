
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.sum()
        return v1
# Inputs to the model
x1 = torch.randn((100, 101), requires_grad=True)
inp = torch.randn((102, 101), requires_grad=False)
