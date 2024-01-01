
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x.T, x.T), dim=1)
        y = x.view(-1).repeat_interleave(2) if x.shape[0] == 1 else y.tanh()
        return y.view(y.shape[0], 4, 4)
# Inputs to the model
x = torch.randn(2, 4, 4)
