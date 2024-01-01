
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        v1 = input1.view(-1, input1.shape[0])
        v2 = input2.view(-1, input2.shape[0])
        v3 = input3.view(-1, input3.shape[0])
        v4 = input4.view(input4.shape[0], -1)
        v5 = torch.matmul(v1, v2)
        v6 = torch.matmul(v3, v4)
        return v5 + v6
# Inputs to the model
x1 = torch.randn(100, 100)
x2 = torch.randn(100, 100)
x3 = torch.randn(100, 100)
x4 = torch.randn(100, 100)
