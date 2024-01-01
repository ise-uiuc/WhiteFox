
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        w1 = torch.tensor([[1.0, 1.0],])
        b1 = torch.tensor([1.0,])
        self.linear = torch.nn.Linear(2, 2, bias=True)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(w1)
            self.linear.bias = torch.nn.Parameter(b1)
 
    def forward(self, x1):
        x2 = self.linear(x1)
        x3 = x2 - other (You should replace 'other' with proper value)
        x4 = F.relu(x3)
        return x4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.tensor([[1.0, 1.0],])
