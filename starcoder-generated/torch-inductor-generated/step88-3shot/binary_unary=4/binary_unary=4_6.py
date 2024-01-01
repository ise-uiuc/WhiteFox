
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x1, other=None):
        x2 = self.linear(x1)
        return x2 + other if other is not None else x2

# Initializing the model
m = Model()

output = m(x1, x2) if x2 is not None else m(x1)
__output1__ = output + 1
__output2__ = output * 2
__output3__ = output >> 1
__output4__ = output.clamp(0.0, 8.0)
__output5__ = output.clone()
__output6__ = output.detach()
