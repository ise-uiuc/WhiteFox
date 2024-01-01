
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
 
    def forward(self, v_in, v_other=None):
        t1 = self.linear(v_in)
        if v_other is None:
            result = t1
        else:
            t2 = t1 + v_other
            result = t2
        return result

# Initializing the model
m = Model()
 
# Inputs to the model
v_in = torch.randn(1, 1)
v_other = torch.randn(1, 1)

# Inputs to the model
