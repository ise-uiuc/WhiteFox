
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 1)
 
    def forward(self, x):
        v1 = self.linear1(x)
        v2 = v1.transpose(-2, -1)
        v2 = v1.div(math.sqrt(v1.shape[-1]))
        v2 = torch.nn.functional.softmax(v2, -1)
        v3 = self.linear2(v2)
        return v3 * v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 2)
