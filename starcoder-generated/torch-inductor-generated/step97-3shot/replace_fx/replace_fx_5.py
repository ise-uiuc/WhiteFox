
class Model(torch.nn.Module):
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        return x2
# Inputs to the model
x1 = torch.randn([1, 2, 2])
