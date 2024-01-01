
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.view(-1)
        l1 = v1.dot(w)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        v2 = l5.view(x1.shape)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
w = torch.randn((x1.shape[1]))
