
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 3)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
m = Model()

# Inputs to the model, 'x2' (referred to as 'other' in the model) should be a constant tensor
x1 = torch.rand(2, 5, requires_grad=True)
x2 = torch.tensor([4.7, 7.9, -8.1])
