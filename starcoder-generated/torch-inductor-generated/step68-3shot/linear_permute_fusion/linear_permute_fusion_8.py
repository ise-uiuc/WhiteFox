
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 5)
    def forward(self, x1):
        v2 = torch.cat((x1, self.linear.weight, self.linear.bias.view(1, 5)), dim=1)
        v1 = v2.permute(1, 0)
        return v1
# Inputs to the model
x1 = torch.ones(2, 5)
