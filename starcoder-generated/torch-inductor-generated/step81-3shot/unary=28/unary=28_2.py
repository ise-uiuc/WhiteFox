
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32, 1, bias=True)
 
    def forward(self, x2, param_min=-0.35, param_max=0.35, ):
        v1 = self.linear(x2)
        v2 = torch.clamp_min(v1, param_min)
        v3 = torch.clamp_max(v2, param_max)
        return v3
 
# Initializing the model
m = Model()

# Input to the model
x2 = torch.randn(1, 16)
