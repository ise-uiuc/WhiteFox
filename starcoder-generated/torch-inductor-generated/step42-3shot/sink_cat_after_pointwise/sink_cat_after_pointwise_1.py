
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x.permute(3, 2, 1, 0), x), dim=0)
        y = y + y
        return y.view(-1).tanh() if y.shape!= (8, 2, 3, 1) else y.view(-1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 1, 2)
