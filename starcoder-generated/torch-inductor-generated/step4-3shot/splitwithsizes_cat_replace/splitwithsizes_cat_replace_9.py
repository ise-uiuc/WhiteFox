
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        t1 = torch.split(x1, [16, 16, 16], dim=2) # Split the input tensor with size (1, 3, 64, 64) into 3 tensors along the dimension with size 16
        t2 = torch.cat(t1, dim=2) # Concatenate the split tensors along the dimension with size 16, i.e. with (1, 3, 48, 64)
        x2 = t2 * 0.5
        x3 = t2 * 0.7071067811865476
        x4 = torch.erf(x3)
        x5 = x4 + 1
        x6 = x2 * x5
        return x6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
