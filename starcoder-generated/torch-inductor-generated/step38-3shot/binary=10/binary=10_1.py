
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 + x2
        return v2

# Initializing the model
m1 = Model()

# Generating the input tensors
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)

# Outputs from the model
__output1__ = m1(x1, x2)

__output2__ = m1(x1, m1(x1, x2))

__output3__ = m1(x1, m1(x1, m1(x1, x2)))

m2 = Model()
__output4__ = m2(torch.zeros_like(x1), x2)
