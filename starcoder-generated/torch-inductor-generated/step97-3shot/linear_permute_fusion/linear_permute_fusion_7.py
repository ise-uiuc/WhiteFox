
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x2):
        v0 = x2
        v1 = torch.addmm(mat1=v0, mat2=v0, mat=self.linear.weight, beta=1, alpha=1)
        v2 = torch.tanh(v1)
        v3 = v2.permute(1, 0)
        v4 = v3.view(2, 2)
        return v4
# Inputs to the model
x2 = torch.randn(1, 2, 2)
