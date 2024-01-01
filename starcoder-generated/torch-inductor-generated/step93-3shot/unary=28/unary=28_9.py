
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10, bias=False)
        self.__constants__ = [min_value, num_value]
 
    def forward(self, x):
        v1 = self.linear(x)
        v2 = F.clamp_min(v1, self.__constants__[0])
        v3 = F.clamp_max(v2, self.__constants__[1])
        return v3


# Initializing the model
m = Model(torch.tensor(0., device='cpu'), torch.tensor(1., device='cpu'))

# Input to the model
x = torch.randn(1, 10, device='cpu')
