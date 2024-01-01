
class Model(torch.nn.Module):
    def __init__(self, x_num):
        super().__init__()
        self.linear = torch.nn.Linear(x_num, x_num)
 
    def forward(self, x1):
        y1 = self.linear(x1)
        y2 = torch.sigmoid(y1)
        y3 = y2 * y1
        return y3

# Initializing the model
x_num = 5
m = Model(x_num)

# Inputs to the model
x1 = torch.randn(1, x_num)
