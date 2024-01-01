 (with extra input)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x_cat_dim_0):
        v1 = torch.cat((x1, x2), dim=0)
        v2 = torch.cat((x1, x2), dim=x_cat_dim_0)
        return v1, v2
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(2, 3)
