
class Model(torch.nn.Module):
    def forward(self, inp):
        input_shape = torch.split(torch.split(inp, 16, dim=3)[0], 16, dim=2)
        return torch.stack(input_shape, dim=2)
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
