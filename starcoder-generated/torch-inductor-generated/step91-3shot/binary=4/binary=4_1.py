
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(in_features=128, out_features=128, bias=True)
 
    def forward(self, x1, x2, scale, shift):
        v1 = self.sigmoid(max(-15, min(x1, 15)))
        v2 = self.sigmoid(max(-15, min(x2, 15)))
        v3 = self.addmm(v1, v2, self.linear.weight, beta=1, alpha=scale)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
x2 = torch.randn(2, 128)
scale = torch.randn(128)
shift = torch.randn(128)
