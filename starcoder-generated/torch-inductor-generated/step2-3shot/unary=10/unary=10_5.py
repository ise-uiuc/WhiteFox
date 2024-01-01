
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for module in self.modules():
            if type(module) == torch.nn.Linear:
                module.bias.data = module.bias.data / 6
                module.weight.data = module.weight.data / 6
 
    def forward(self, x1):
        l1 = torch.nn.functional.linear(x1, torch.nn.Linear(3, 8, bias=False).weight)
        l2 = l1 + 3
        l3 = torch.nn.functional.relu6(l2)
        return l3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
