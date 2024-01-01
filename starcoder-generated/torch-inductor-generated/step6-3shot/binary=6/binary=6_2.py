
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.nn.functional.batch_norm(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=1.1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 4, 4, requires_grad = True)
__tensor_input__ = x1
