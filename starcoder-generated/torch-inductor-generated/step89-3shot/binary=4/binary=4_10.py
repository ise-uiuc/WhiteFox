
class Model(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = torch.nn.Linear(5, num_classes)
 
    def forward(self, x1, other1):
        v1 = self.linear(x1)
        v2 = v1 + other1
        return v2

# Initializing the model
m=Model(3)

# Inputs to the model
x1=torch.randn(1, 5)
__output__=m(x1, x1)

