
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v4 = x1
        a1 = self.relu(v4)
        v1 = torch.nn.functional.softmax(a1)
        v2 = a1.permute(0, 1, 3, 2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu', requires_grad=True)
