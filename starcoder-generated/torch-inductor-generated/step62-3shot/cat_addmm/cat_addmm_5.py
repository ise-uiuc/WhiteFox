
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.cpu()
        y = x.addmm(torch.ones(5), torch.ones(5))
        z = y.type_as(y.new_ones(5))
        x = x.new_zeros(2)
        y = z * z
        y = y.cpu()
        return y
# Inputs to the model
x = torch.rand(5)
z = torch.rand(1,5)
