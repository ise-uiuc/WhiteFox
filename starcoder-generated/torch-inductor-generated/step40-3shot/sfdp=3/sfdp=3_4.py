
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1, x2, x3):
        v1 = torch.matmul(x1, x2.transpose(-2, -1))
        v2 = v1 * 0.0625
        v3 = torch.nn.functional.softmax(v2, dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.10000000149011612, training=self.training, inplace=False)
        v5 = torch.matmul(v4, x3)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128, 768)
x2 = torch.randn(1, 768, 128)
x3 = torch.randn(1, 128, 128)
