
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        t1 = input.permute(0, 2, 1)
        v1 = self.linear(t1)
        return v1
# Inputs to the model
x = torch.randn(1, 2, 2)
