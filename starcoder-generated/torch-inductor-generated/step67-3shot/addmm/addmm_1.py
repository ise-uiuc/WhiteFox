
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Please add extra random tensors here
        self.tensor1 = torch.randn(3, 3, requires_grad=True)
        self.tensor2 = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x1)
        v1 = torch.mm(v1, x1)
        v1 = v1 + self.tensor1
        # Add several operators here
        v2 = v1 + self.tensor2 + torch.mm(v1, self.tensor1) + torch.mm(self.tensor2, v1)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
