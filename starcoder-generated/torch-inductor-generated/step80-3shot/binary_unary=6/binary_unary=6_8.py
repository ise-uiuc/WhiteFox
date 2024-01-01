
class Model(torch.nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.lin = torch.nn.Linear(in_feat, out_feat)
 
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = v1 - 10
        v3 = torch.nn.ReLU()(v2)
        return v3

# Initializing the model
m = Model(28 * 28, 50)

# Inputs to the model
x1 = torch.randn(5, 28, 28)
