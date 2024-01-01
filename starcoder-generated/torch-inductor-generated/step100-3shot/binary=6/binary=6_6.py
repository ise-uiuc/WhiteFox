
class Model(torch.nn.Module):
    __linear = torch.nn.Linear(8, 8)
 
    def __init__(self):
        super().__init__()
        for name, v in self.__linear.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.orthogonal_(v)
 
    def forward(self, x1):
        v1 = self.__linear(x1)
        v2 = v1  - 5
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
