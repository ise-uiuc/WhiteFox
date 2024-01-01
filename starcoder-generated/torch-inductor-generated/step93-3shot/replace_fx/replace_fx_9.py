
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # This pattern requires differentiable inputs.
        # See https://discuss.pytorch.org/t/how-do-i-generate-differentiable-inputs-eg-for-autograd-tests/15378
        self.x1 = torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))
    def forward(self, x2):
        t1 = torch.rand_like(x2)
        x3 = self.x1 + t1
        return x3
# Inputs to the model
x2 = torch.randn(2, 2, requires_grad=True)
