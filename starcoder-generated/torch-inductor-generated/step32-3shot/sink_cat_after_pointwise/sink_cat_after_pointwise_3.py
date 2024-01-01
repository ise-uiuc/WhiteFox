
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x.transpose(2, 3)), dim=1)
        return y.reshape(x.shape[0], -1).tanh() if y.shape!= (1, 3) else y.reshape(x.shape[0], -1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
