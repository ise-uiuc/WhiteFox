
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = torch.tensor([[0.7631, 0.9574, 0.5303], [0.3785, 0.5842, 0.4723], [0.6486, 0.6524, 0.4806]])
        v2 = torch.tensor([[-0.2000, 0.7019, 1.0966], [0.5140, 0.9784, 0.8893], [1.4617, 1.9182, 1.4599]])
        v3 = torch.addmm(x1, v1, v2)
        v4 = v3.clamp(min=0, max=5)
        v5 = torch.cat([v4])
        v6 = v5 + 0.42023874638647563
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
