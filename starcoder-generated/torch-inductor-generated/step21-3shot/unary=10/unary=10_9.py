
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        b1 = torch.flatten(x1)
        d1 = b1[..., np.newaxis]
        l1 = torch.nn.functional.linear(d1, 2, bias=None) # Applies a linear transformation to the input tensor with zero bias
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100, 100)
