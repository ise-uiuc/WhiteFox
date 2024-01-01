
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        x1_tmp = torch.cat([x1, x2], dim=1)
        x2_cat_tmp = x1_tmp[:, -1]
        x3_cat = torch.cat([x2_cat_tmp, x3], dim=1)
        return x3_cat

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 1)
