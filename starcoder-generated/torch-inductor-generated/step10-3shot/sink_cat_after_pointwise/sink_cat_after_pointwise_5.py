
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        for i in range(0, 2):
            y = y.tanh()
            if y.shape[0] == 1:
                y = torch.cat((y, y), dim=1)    
        for i in range(0, 2):
            y = k * y.view(x.shape[0], -1).tanh()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
