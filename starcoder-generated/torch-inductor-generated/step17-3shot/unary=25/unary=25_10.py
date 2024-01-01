
```class Model(torch.nn.Module):
    def __init__(self, n1, n2, neg_slope1, neg_slope2):
        super().__init__()
        self.linear = torch.nn.Linear(n1, n2)
        self.neg_slope1 = neg_slope1
        self.neg_slope2 = neg_slope2
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.neg_slope1
        v4 = torch.where(v2, v1, v3)
        v5 = v1 * self.neg_slope2
        v6 = torch.where(v2, v4, v5)
        return v6```

# Initializing the model
m = Model(3, 1, 0.01, 0.03)

# Inputs to the model
x1 = torch.randn(1, 3)
