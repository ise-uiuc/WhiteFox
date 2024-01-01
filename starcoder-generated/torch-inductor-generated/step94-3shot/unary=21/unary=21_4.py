
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.randn(3, 22, 22, 2)
        v2 = x.permute(0, 3, 1, 2)
        v3 = torch.einsum('i...j, ijkl -> i...kl', v1, v2)
        return v3
# Inputs to the model
tensor = torch.randn(7, 3, 22, 22)
