
class SinkRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.randn(2)
    def forward(self, x):
        x = torch.relu(torch.cat((x, x, torch.abs(self.a), self.a), dim=1).view(x.shape[0], -1).view(x.shape[0], -1) + torch.cat((torch.cat((self.a, x), dim=1), self.a), dim=1).view(x.shape[0], -1) + x)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
