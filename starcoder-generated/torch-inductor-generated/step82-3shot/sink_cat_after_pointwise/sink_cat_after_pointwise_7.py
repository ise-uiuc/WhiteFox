
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(3):
            z = torch.nn.functional.tanh(y.clone().view(y.shape[0], -1))
            y = torch.cat([y, z], dim=1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 32, 4)
