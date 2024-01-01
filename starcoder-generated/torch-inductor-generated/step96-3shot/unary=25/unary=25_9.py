
class Model_LQ_ReLU(torch.nn.Module):
    def __init__(self, negative_slope=0.1):
        super(Model_LQ_ReLU, self).__init__()
        self.linear = torch.nn.Linear(10, 16)
        self.negative_slope = negative_slope
 
    def forward(self, x):
        x = x.view(-1,10)
        x = self.linear(x)
        x = x > 0
        x = x * self.negative_slope
        x = torch.where(x, x, x*self.negative_slope)
        return  x

# Initializing the model
m = Model_LQ_ReLU()

# Inputs to the model
x = torch.randn(1, 10)
