
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x2):
        v2 = torch.nn.functional.linear(x2, weight=__weight1__, bias=__bias1__)
        v3 = torch.clamp_min(v2, min=__minvalue__)
        v4 = torch.clamp_max(v3, max=__maxvalue__)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 32)
