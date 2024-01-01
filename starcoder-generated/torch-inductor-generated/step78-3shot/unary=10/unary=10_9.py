
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8192, 10)
 
    def forward(self, x1):
        # l1 = linear(input_tensor) 
        o1 = self.linear(x1)
        # l2 = l1 + 3
        o2 = o1.add(3)
        # l3 = torch.clamp_min(l2, 0) 
        o3 = torch.clamp_min(o2, 0)
        # l4 = torch.clamp_max(l3, 6) 
        o4 = torch.clamp_max(o3, 6)
        # l5 = l4 / 6
        o5 = o4.div(6)
        return o5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8192)
