
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        x = x1
        l1 = x.transpose(1, 2).contiguous().view(-1, x1.size(1))
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        x = l5.view(x1.size(0), x1.size(2), x1.size(3), x1.size(1)).transpose(1, 3).contiguous()
        return x

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
