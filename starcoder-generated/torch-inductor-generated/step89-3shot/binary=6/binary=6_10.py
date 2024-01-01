
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
 
    def forward(self, input):
        out = self.linear(input)
        out = out - torch.mean(self.linear.weight, 1, keepdim=True)
        out = out - torch.mean(self.linear.bias)
        return out
 
# Initializing the model
m = Model()
 
# Inputs to the model
input = torch.randn(10)
