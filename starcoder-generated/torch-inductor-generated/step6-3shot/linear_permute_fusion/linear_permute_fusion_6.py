
class Model(torch.jit.ScriptModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)
    @torch.jit.script_method
    def forward(self, x1):
        v4 = x1
        v1 = torch.nn.functional.linear(v4, self.linear.weight, self.linear.bias)
        v2 = v1.permute(0, 2, 1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
