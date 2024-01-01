
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.c1 = nn.ConvTranspose2d(127, 41, (11, 10), 1, (5, 4), 1)
    def forward(self, input13):
        input12 = self.c1(input13)
        input15 = input12 > 0
        input16 = input12 * -1.13
        input14 = torch.where(input15, input12, input16)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.relu(input14), (5, 3))
# Inputs to the model
input13 = torch.randn(684, 127, 25, 9)
