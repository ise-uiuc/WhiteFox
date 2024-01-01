
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, ((x-1.)/10), (-x/130) + (x-1.)/100, (x-1.)/100, x*(-1./10) + 1./100), dim=1)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2,1)
