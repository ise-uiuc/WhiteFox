
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(512, 256, bias=False)
 
    def forward(self, x1): 
        v1 = self.linear(x1)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 512, 7, 7)
