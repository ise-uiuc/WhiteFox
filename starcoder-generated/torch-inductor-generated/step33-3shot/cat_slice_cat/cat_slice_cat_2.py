
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, *input_tensors):
        t1 = torch.cat(input_tensors, dim=1)
        t2 = t1[:, -1]
        t3 = t2[:, 112]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 3)
x3 = torch.randn(1, 4)
