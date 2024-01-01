
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        if y.shape!= (5, 2):
            y = torch.relu(y)
        return y.view(-1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
