
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = torch.sqrt_(torch.FloatTensor([math.sqrt(512)]))
 
    def forward(self, x1, x2):
        s1 = torch.matmul(x1, x2.transpose(-2, -1))
        s2 = s1 * self.scale_factor
        s3 = torch.nn.functional.softmax(s2, dim=-1)
        d1 = torch.nn.functional.dropout(s3)
        d2 = torch.matmul(d1, x2)
        return d2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
