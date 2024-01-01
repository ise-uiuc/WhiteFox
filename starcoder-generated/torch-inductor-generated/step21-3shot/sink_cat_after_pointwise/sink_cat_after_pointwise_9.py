
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(-1)
        if torch.jit.is_scripting():
            y = y.view(5, 3)
        else:
            y = y.view(5, 3).tanh()
        y = torch.cat((y, y), dim=0)
        y = y.relu()
        y = y.clamp(min=0, max=10)
        y = y.tanh()
        x = x.tanh()
        return y + x
# Inputs to the model
x = torch.randn(5, 5)
