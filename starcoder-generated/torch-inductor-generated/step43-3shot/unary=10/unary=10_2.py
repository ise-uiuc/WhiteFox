
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8,1)
 
    def forward(self, x1):
        q1 = self.linear(x1)
        q2 = q1 + 3
        q3 = torch.clamp_min(q2, 0)
        q4 = torch.clamp_max(q3, 6)
        q5 = q4 / 6
        return q5

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 8)
