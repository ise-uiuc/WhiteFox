
class Model(torch.nn.Module):
    def __init__(self):
        self.linear = torch.nn.Linear()
        print("init done 1")
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 3.1415926
        v3 = torch.relu(v2)
        print("done 2")
        return v3

# Initializing the model
m = Model()
# Inputs to the model
x1 = torch.randn(1024,500)
