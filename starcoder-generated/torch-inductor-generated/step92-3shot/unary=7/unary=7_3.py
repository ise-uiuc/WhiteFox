
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=128, out_features=256, bias=True)
 
    def forward(self, x1):
        l1 = self.linear(x1)
        l2 = l1 * torch.clamp(l1+3, min=0, max=6)
        return l2 / 6

# Initializing the model
m = Model()

# Input to the model. Please make sure the shape of the tensor is correct.
x1 = torch.randn(1, 128)
