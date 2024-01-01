
class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, y) + torch.mm(x, y) # Matrix multiplication
# Inputs to the model
x = torch.randn(20, 5)
y = torch.randn(5, 40)
