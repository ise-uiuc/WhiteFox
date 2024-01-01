
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = torch.nn.Linear(10, 10, bias=False)
 
    def forward(self, x1, x2):

        return self.linear(x1) + x2


# Initializing the model
m = Model() 

other = torch.randn(1, 10)
x1 = torch.randn(10)
x2 = torch.randn(10)

# Inputs to the model
