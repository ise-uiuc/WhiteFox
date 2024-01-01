
class Model(torch.nn.Module):
    def __init__(self, W1: torch.Tensor, W2: torch.Tensor):
        super().__init__()
        self.W1 = torch.nn.Parameter(W1)
        self.W2 = torch.nn.Parameter(W2)
    def forward(self, input1, input2):
        h1 = torch.mm(self.W2, input1)
        h2 = torch.mm(input2, self.W1)
        return h1 + h2
# Inputs to the model
W1 = torch.randn(64, 64)
W2 = torch.randn(64, 64)
input1 = torch.randn(64, 64)
input2 = torch.randn(64, 64)
