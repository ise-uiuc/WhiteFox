
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x3)
        v2 = torch.matmul(x1, x3)
        v3 = torch.matmul(x1, x3)
        return v1 + v2, v3
# Inputs to the model
batch_size = 2
input1 = torch.randn(batch_size, 14, 8)
input2 = torch.randn(batch_size, 8, 16)
input3 = torch.randn(batch_size, 16, 7)
