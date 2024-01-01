
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        o1 = x1.view(x1.shape[0], -1)
        o2 = F.linear(o1, 0.25, 0.25)
        o3 = F.selu(o2)
        o4 = o3 + 1.6732632423543772848170429916717
        o5 = o4 * 0.25
        o6 = o5 * 0.25
        return o6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1000)
