
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        def func(x, y):
            return x + y
        class Module(torch.nn.Module):
            def forward(self):
                return self.linear(input)
        m = Module().train()
        y = m.forward()
        return func(y, x) 
# Inputs to the model
input = torch.rand(3, 4)
