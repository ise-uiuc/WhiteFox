
class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 48, bias=False)
        def forward(self, input):
            v1 = self.act(self.linear(input))
            v2 = v1 + 3
            v3 = torch.clamp_min(v2, 0)
            v4 = torch.clamp_max(v3, 6)
            return(v4 / 6)

 # Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1,8)
