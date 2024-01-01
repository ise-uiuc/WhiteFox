
class Model(torch.nn.Module):
    def forward(self, x):
        result = [torch.ops.aten.addmm(x, x, x)] * 5
        x = torch.aten.cat(result, 0)
        return x

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(3, 3)
