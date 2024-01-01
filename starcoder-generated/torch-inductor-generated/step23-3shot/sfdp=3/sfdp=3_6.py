
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
 
    def forward(self, x1, x2, scale_factor=1 / math.sqrt(512)):
        x3 = self.softmax((torch.matmul(x1, x2.transpose(-2, -1)).mul(scale_factor)))
        x4 = torch.nn.functional.dropout(x3, p=0.1)
        output = x4.matmul(x2)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10, 512)
x2 = torch.randn(1, 10, 512)
