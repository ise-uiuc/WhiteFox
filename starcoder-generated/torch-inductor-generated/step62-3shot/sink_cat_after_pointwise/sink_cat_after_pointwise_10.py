
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x.T, x.T), dim=1).view(-1) if x.shape[0] == 1 else x.tanh()
        return y
# Inputs to the model
x = torch.randn(12, 4)
