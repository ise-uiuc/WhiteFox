
class SinkRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.tensor(0.0)
        self.in_features = 8
        self.out_features = 12
    def forward(self, x):
        # return torch.add(torch.relu(x.view(-1)), x.view(-1))
        return torch.add(torch.relu(x.view(x.shape[0], -1)), x.view(x.shape[0], -1))
# Inputs to the model
x = torch.randn(3, 8, requires_grad=True)
