
class Model(torch.nn.Module):
    def forward(self, x, w):
        return torch.cat((x * w, x), 1).view(-1)

# Weight tensor
w = torch.randn(1, 2)

# Inputs to the model
x = torch.randn(2)

