
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cat_dim = 1
 
    def forward(self, inputTensorList, index):
        x1, x2, x3, x4 = x
        v0 = torch.cat([x1, x2], dim=self.cat_dim)
        v1 = torch.cat([x3, x4], dim=self.cat_dim)
        v2 = v0[:, :index]
        v3 = v2[:, :index]
        v4 = torch.cat([v1, v3], dim=self.cat_dim)
        return result

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(4, 32, 256, 256)
index = 20
