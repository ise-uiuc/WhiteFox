
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x_A, x_B):
        temp_1 = torch.cat([x_A, x_B], dim=2)
        temp = temp_1.permute(0, 2, 1)
        return torch.matmul(temp, temp_1)
# Inputs to the model
x_A = torch.randn(1, 2, 2)
x_B = torch.randn(1, 2, 2)
