
class Model(torch.nn.Module):
    def forward(self, input1, input2, scale, dp):
        v1 = torch.matmul(input1, input2.transpose(-2, -1))
        v2 = v1.mul(scale)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=dp)
        v5 = v4.matmul(input2)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
input1 = torch.randn(4, 8, 5)
input2 = torch.randn(4, 7, 8)
scale = torch.tensor([1.0])
dp = 0.5
