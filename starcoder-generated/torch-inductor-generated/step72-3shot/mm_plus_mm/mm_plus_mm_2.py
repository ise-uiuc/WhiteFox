
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weight = torch.randn(1, 5)
    def forward(self, inputs):
        out = torch.div(torch.matmul(inputs, self.weight), 2)
        out = torch.div(out, 2)
        return out
# Inputs to the model
inputs = torch.randn(5, 5)
