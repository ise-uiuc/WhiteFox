
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=0)
        x = y.view(-1).tanh() 
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
