
class Model(torch.nn.Module):
    pass
 
# Initializing the model
m = Model()
 
# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 64)
x3 = torch.randn(1, 5)
x4 = torch.randn(1, 4)
__input_tensor__ = [
    x1,
    x2,
    x3,
    x4
]
