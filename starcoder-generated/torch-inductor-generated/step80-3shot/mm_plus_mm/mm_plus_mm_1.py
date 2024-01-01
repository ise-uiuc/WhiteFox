
class Model(torch.nn.Module):
    def __init__(self, weight):
        super(Model, self).__init__()
        self.weight = weight
    def forward(self, input):
        t1 = torch.mm(self.weight, self.weight)
        t2 = torch.mm(self.weight, self.weight)
        t3 = torch.mm(self.weight, self.weight)
        return t1 + t2 + t3
# Inputs to the model
weight = torch.randn(10, 10)
input = torch.randn(10, 10)
