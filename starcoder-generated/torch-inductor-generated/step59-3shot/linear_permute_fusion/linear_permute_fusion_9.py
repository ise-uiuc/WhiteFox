
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, input):
        v7 = self.linear(input)
        v0 = input.permute(0, 2, 1).permute(0, 1)
        return v0
# Inputs to the model
input = torch.randn(1, 2, 2)
