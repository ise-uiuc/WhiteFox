
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x[:, :3], x[:, 4:]], dim=1)
        x = y.reshape(y.shape[0], x.shape[1], -1)
        return x.tanh()
# Inputs to the model
x = torch.randn(2, 10, 4)
