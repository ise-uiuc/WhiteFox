
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x2):
        l1 = x2.reshape(64 * 64, 3)
        l2 = l1 + 3
        l3 = torchvision.ops.misc.clamp_min(l2, 0)
        l4 = torchvision.ops.misc.clamp_max(l3, 6) 
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
