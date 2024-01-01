
class Model(torch.nn.Module):
    def __init__(self, neg_slope):
        super().__init__()
        self.fc = torch.nn.Linear(8, 256)
        self.neg_slope = neg_slope
 
    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1.shape[-1]
        t1 = v1.reshape([-1, v2])
        v3 = t1 > 0
        v6 = v1.shape[-1]
        t2 = v3.reshape([-1, v6])
        v4 = t1 * self.neg_slope
        v5 = torch.where(t2, v1, v4)
        t3 = v5.reshape([-1, 8, 32, 32])
        return t3

# Initializing the model
m = Model(0.1)

# Inputs to the model
x1 = torch.randn(1, 8, 32, 32)
