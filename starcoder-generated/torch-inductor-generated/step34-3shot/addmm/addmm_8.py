
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v3 = v1.add(x1, alpha=1) # This pattern only works with torch.nn.functional.interpolate. 
                                 # However, it works for torch.cat, torch.mm, torch.bmm, and torch.mv.
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
