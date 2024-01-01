
class Model_customization(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2):
        matmul_result = torch.mm(input1, input2)
        mul1 = matmul_result * 2
        mul2 = mul1 * 2
        return mul2
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3, requires_grad=True)
