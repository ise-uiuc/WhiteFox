
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(3072, 8)
 
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = v1 - 0.5
        return v2
   
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3072)
