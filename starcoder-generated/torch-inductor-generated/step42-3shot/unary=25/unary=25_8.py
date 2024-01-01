
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, weight=torch.zeros((x1.shape[1], x1.shape[1])))
        v2 = v1 > 0
        v3 = v1 * 0.01
        v4 = torch.where(v2, v1, v3)
        return v4
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
