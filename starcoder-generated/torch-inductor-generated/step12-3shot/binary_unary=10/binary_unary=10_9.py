
class Model(torch.nn.Module):
    def __init__(self, out_channel):
        super(Model, self).__init__()
        self.out_channel = out_channel
        self.linear = torch.nn.Linear(10, out_channel)
 
    def forward(self, x1, x2):
        t1 = self.linear(x)
        t2 = t1 + x2
        t3 = torch.nn.functional.relu(t2)
        return t3
 
# Initializing the model
m = Model(8)
 
# Inputs to the model
x1 = torch.randn(5, 10)
x2 = torch.randn(5, 8)
