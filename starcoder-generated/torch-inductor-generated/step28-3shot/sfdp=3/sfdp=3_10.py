
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.125
        v3 = v1 * 0.20884964425119184
        v4 = v3.softmax(dim=-1)
        v5 = torch.nn.functional.dropout(v4, p=0.125)
        output = v5.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 8, 4, 4)
x2 = torch.randn(2, 8, 10, 20)
