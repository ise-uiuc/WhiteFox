
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        y = torch.cat((y, y), dim=1)
        y = y.view(y.shape[0], -1)
        y = y.tanh()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
# Model Ends
