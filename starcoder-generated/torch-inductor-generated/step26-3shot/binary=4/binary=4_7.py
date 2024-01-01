
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        e1 = torch.nn.CrossEntropyLoss(reduction="mean")
        v2 = e1(v1, x2)
        v3 = v2 + 1
        v4 = torch.exp(v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 10)
x2 = torch.tensor([1, 2])
