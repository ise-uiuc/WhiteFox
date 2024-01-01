
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        q1 = self.conv(x1)
        k1 = self.conv(x2)
        dot_1 = torch.matmul(q1, k1)
        v1 = dot_1.div(0.0001)
        softmax_1 = v1.softmax(dim=-1)
        dropout_1 = torch.nn.functional.dropout(softmax_1, p=0.1)
        output = dropout_1.matmul(v1)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
