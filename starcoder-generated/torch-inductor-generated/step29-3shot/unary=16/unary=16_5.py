
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.nn.functional.relu(v1)
        return v2
        
# Initializing the model
torch.manual_seed(1234)
m = Model()

# Inputs to the model
torch.manual_seed(1234)
x1 = torch.randn(1, 10)
