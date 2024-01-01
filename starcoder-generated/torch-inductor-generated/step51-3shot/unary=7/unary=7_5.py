
def _scaled_linear(scale, x):
    with torch.no_grad():
        alpha = scale /( 1 + torch.abs(x))
        alpha[x > 0] = scale
        alpha[x < 0] = 0
    return x * alpha
     
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
       v1 = torch.nn.functional.linear(x, x)
       v2 = _scaled_linear(min=0, max=6, x=v1 + 3)
       v3 = v2 / 6
       return v3

# Initializing the model
m = Model()
 
# Input to the model
x = torch.randn(2, 3)
