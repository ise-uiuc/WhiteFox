
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 32
        self.out_features = 8
        self.weight = torch.randn(self.out_features, self.in_features)
        self.bias = torch.randn(self.out_features)
 
    def forward(self, x1):
        l1 = F.linear(x1, self.weight, self.bias)
        l2 = l1 + 3
        l3 = torch.clamp_min(l2, 0)
        l4 = torch.clamp_max(l3, 6)
        l5 = l4 / 6
        return l5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
