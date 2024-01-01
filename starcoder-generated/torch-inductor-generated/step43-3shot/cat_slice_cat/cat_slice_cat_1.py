
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        cat = [x1, x2]
        t1 = torch.cat(cat, 1)
        t2 = t1[:, 0:255]
        res = t2[:, 0:10]
        return res

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 255, 255)
x1 = x1 - 50
x2 = torch.randn(10, 255, 255)
x2 = x2 - 50
