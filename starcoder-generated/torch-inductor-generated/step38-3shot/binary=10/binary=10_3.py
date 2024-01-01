
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)
 
    def forward(self, __x1__):
        v1 = self.linear(__x1__)
        v2 = v1 + __other__
        return v2

# Initializing the model
m = Model()


# Input to the model
x1 = torch.randn(1, 3, 64, 64)

