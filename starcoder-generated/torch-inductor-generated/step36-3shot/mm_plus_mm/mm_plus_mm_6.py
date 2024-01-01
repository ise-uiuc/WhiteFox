
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(60, 40)
        self.linear2 = torch.nn.Linear(40, 20)
        self.linear3 = torch.nn.Linear(20, 1)
    def forward(self, inputs, weight1, weight2, weight3):
        v1 = self.linear1(inputs)
        v2 = self.linear2(inputs)
        v3 = self.linear3(inputs)
        return v1 + v2 + v3
# Inputs to the model
inputs = torch.randn(1, 60)
weight1 = torch.randn(20, 60)
weight2 = torch.randn(20, 40)
weight3 = torch.randn(20, 20)
