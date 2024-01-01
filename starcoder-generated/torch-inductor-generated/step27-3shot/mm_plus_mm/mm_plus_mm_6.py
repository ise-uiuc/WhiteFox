
class Net(nn.Module):
    def forward(self, x1, x2):
        return -torch.matmul(x1, x2)
# Inputs to the model
w = torch.randn(3, 3)
x1 = torch.randn(4, 5)
x2 = torch.randn(5, 7)
