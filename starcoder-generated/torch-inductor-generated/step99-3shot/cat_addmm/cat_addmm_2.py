
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 3)
        self.linear2 = torch.nn.Linear(8, 8)
        self.linear3 = torch.nn.Linear(5, 3)
        self.concat = torch.cat
    def forward(self, input):
        x = self.linear1(input)
        y = self.linear2(input)
        z = self.linear3(input)
        output = self.concat([x,y,z], dim=2)
        return output
# Inputs to the model
x = torch.randn(2, 8)
