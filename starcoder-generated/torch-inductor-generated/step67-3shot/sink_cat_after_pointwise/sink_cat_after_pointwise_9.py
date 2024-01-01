
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if not torch.equal(x, torch.zeros(x.shape)):
            x = torch.cat((x, torch.zeros(x.shape)), dim=0)
        x *= 2
        return x
# Inputs to the model
x = torch.randn(3, requires_grad=True)
