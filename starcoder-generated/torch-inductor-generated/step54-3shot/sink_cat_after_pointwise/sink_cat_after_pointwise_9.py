
class Model(torch.nn.Module):
    def forward(self, x):
        return x.view(-1).view(x.shape)
# Inputs to the model
x = torch.randn(2, 3, 4)
