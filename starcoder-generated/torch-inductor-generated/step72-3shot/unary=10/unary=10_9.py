
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        s1 = self.linear(x1)
        s2 = s1 + 3
        s3 = torch.clamp_min(s2, 0)
        s4 = torch.clamp_max(s3, 6)
        s5 = s4 / 6
        return s5
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
