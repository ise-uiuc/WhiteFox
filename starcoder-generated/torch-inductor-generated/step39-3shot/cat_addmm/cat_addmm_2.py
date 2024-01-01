
class Model(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.addmm(input=x, mat1=torch.randn(2, 2), mat2=torch.randn(2, 2))
        return x
# Input to the model
x = torch.randn(1, 2)
