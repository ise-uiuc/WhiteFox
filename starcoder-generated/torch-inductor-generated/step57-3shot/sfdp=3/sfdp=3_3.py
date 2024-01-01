
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x, y, scale_factor, dropout_p):
        v1 = torch.matmul(x, y.transpose(-2, -1))
        v2 = v1 * scale_factor
        v3 = torch.nn.functional.softmax(v2, -1)
        v4 = torch.nn.functional.dropout(v3, p=dropout_p)
        v5 = torch.matmul(v4, y)
        return v5

# Inputs to the model
x = torch.randn(1, 256, 2048)
y = torch.randn(1, 2048, 512)
scale_factor = torch.tensor([0.1])
dropout_p = 0.5
