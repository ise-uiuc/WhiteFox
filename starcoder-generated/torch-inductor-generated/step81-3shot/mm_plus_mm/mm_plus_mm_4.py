
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        h1 = torch.matmul(input1, input2)
        h2 = torch.matmul(input2, input1)
        h4 = torch.matmul(input4, input4)
        h5 = torch.matmul(input3, input4)
        h3 = h1 + h2 + h4 + h5
        return h3
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
