
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = torch.nn.functional.split(1, 2)
 
    def forward(self, x2):
        list = self.split(x2)
        v3 = torch.cat([list[1] for i in range(len(list))], 1)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 4, 64, 64) 
