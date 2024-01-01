
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2
# Inputs to the model
x1 = torch.tensor([[1.0]])
x2 = torch.tensor([[3.0, 4.0]])
