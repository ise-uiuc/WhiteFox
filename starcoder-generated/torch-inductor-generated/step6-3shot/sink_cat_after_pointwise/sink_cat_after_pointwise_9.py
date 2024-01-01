
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x_new = x.sigmoid()
        x_new = x + x_new
        return x_new
# Inputs to the model
x = torch.randn(1, 2, 3, 4)
