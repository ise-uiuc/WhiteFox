
class Model(torch.nn.Module):
    def forward(self, x):
        return x.transpose(1, 1)
# Inputs to the model
x = torch.randn(1, 3, 4, 5)
