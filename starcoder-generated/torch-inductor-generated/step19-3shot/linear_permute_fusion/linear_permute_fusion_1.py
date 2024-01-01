
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        v1 = torch.nn.functional.linear(input, self.linear.weight, self.linear.bias)
        v2 = v1.reshape(input.size(0), 1, 2, 1, 1, 2)
        v3 = v2.squeeze(-1)
        v4 = v3.permute(0, 3, 5, 1, 4, 2)
        return v4
# Inputs to the model
input  = torch.rand(1, 1, 2, 2)
model  = Model()
script = torch.jit.script(model)
