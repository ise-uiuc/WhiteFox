
class Model(torch.nn.Module):
    def forward(self, x):
        a = torch.randint(0, 3, (x.size(0),))
        return a
# Inputs to the model
x = torch.randn((2,))
