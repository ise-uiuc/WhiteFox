
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, _input_):
        t1 = torch.cat(_input_, dim=1)
        t2 = t1[:, 0:9223372036854775807]
        t3 = t2[:, 0:_input_.size()]
        t4 = torch.cat([t1, t3], dim=1)
        return t4

# Initializing the model
m = Model()

# Inputs to the model
x = [_input_.clone().detach().requires_grad_(True) for _input_ in [_input_0, _input_1]]
