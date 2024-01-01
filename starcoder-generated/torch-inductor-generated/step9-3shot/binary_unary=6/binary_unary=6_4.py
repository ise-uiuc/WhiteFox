
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 11)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        o1 = v1.detach().numpy()
        return o1
      
# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 10)
