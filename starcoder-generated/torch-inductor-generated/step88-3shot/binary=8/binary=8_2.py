
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 5)
        self.linear2 = torch.nn.Linear(5, 5)
    def forward(self, x1, x2):
        # PyTorch's linear layer does not need bias
        # When addbias=True is specified, the layer will automatically add bias with zeros.
        v1 = self.linear1(x1, addbias=True)
        v2 = self.linear2(x2, addbias=True)
        v3 = torch.add(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(5)
x2 = torch.randn(5)
