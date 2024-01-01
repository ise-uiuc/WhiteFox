
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(3, 8)
 
    def forward(self, input):
        v1 = torch.addmm(input, self.l1.weight, self.l1.bias)
        outputs = [v1]
        return outputs

# Initializing the model
m = Model()

# Inputs to the model
input = torch.randn(1, 3)
