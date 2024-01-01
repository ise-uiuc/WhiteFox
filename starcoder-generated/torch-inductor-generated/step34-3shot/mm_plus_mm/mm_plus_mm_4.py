
class Model(torch.nn.Module):
    def forward(self, input1, weights):
        t1 = torch.mm(input1, weights)
        t2 = torch.mm(t1, weights.t())
        return t2.t()
# Inputs to the model
input1 = torch.randn(4, 4)
# Input 2 is not required. It can contain dummy data in the shape of (200, 1000)
weights = torch.randn(100, 1000)
