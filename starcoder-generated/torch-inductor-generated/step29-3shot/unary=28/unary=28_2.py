
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=5, out_features=7, bias=True)
 
    def forward(self, x2):
        o1 = self.linear(x2)
        o2 = torch.clamp_min(o1, min_value=0.1217971)
        o3 = torch.clamp_max(o2, max_value=1.202422)
        return o3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(4, 5)
