
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        output_one = torch.stack((x, x))
        output_two = torch.matmul(x, x.t())
        output_three = torch.matmul(torch.matmul(x, x.t()), x)
        output_four = torch.cat((output_one, output_two, output_three), dim=1)
        return output_four
# Inputs to the model
x = torch.randn(2, 4)
