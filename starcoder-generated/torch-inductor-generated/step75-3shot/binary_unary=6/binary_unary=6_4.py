
class Model(nn.Module):
    def __init__(self):
  			super().__init__()
        self.linear = nn.Linear(10, 5)
 
    def forward(self, x1):
        x2_3 = self.linear(x1) # The shape of x2_3: 20 x 5
        x2_4 = x2_3 - 1.5
        x2_5 = F.relu(x2_4)
        return x2_5


# Initializing the Model
m = Model()

# Inputs to the Model
x1 = torch.randn(100, 10)
