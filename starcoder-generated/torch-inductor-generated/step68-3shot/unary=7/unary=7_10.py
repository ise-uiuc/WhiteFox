
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 128, bias=False)
 
    def forward(self, l1):
        v1 = self.linear(l1)
        v2 = __output__
        return v2

# Initializing the model
m = Model()

# Inputs to the model
l1 = torch.randn(1, 128)
