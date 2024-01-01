
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[0] == 2:
            x = torch.cat((x,x), dim=0)
        return x.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
