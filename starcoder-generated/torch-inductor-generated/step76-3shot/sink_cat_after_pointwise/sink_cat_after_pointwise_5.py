
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x, x, x, x], dim=0)
        x = y.view(y.shape[0], -1)
        if y.shape!= (6, 33):
            x = torch.relu(x)
        return x.view(y.shape[0], -1) if y.shape!= (6, 33) else x.view(y.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
