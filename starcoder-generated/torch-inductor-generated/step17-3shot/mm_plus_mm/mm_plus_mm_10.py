
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        input1 = torch.nn.functional.conv_transpose3d(data=x1, weight=x2, bias=torch.ones(()), stride=x3, padding=x4)
        return input1
# Inputs to the model
x1 = torch.randn(64, 20, 50, 100)
x2 = torch.randn(20, 16, 3, 3)
x3 = torch.randn(64)
x4 = torch.randn(6)
