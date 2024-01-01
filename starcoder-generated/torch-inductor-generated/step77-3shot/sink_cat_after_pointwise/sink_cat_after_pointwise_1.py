
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(5):
            y = y + y.clone()
            y = torch.cat((y, y, y, y), dim=1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
