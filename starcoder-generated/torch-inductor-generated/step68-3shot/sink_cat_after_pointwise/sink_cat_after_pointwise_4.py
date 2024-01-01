
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x), dim=1) if x.shape!= () else x * x
        z = torch.cat((y, y, y), dim=2) if y.shape!= () else y.sin()
# Inputs to the model
x = torch.randn(2, 3)
