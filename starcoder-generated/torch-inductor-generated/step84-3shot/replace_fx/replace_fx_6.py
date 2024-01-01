
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        d = nn.Dropout()
        return d(x)
# Inputs to the model
x = torch.randn(3, 4, 5)
