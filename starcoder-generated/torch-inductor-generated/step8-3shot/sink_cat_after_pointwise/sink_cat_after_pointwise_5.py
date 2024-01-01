
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(-1, x.shape[0])
        y = torch.cat((y, y), dim=1)
        x = y.view(-1, x.shape[0]).tanh() if y.shape[0] == 1 else y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
