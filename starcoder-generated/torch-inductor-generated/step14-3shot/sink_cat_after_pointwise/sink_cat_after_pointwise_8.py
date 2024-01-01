
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        t0 = torch.add(inputs, inputs)
        s = inputs.size(0)
        d = inputs.size(1)
        t1 = torch.add(t0, inputs)
        shape = (s / d, d)
        t2 = torch.reshape(t1, shape)
        out = torch.tanh(t2)
        return out
# Model and inputs
x = torch.randn(20, 30)
