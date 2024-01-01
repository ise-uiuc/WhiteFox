
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1
        self.dropout_p = 0.2
 
    def forward(self, x, y):
        output = torch.matmul(x, y.transpose(-2, -1))
        output = output * self.scale_factor
        output = output.softmax(dim=-1)
        output = torch.nn.functional.dropout(output, p=self.dropout_p)
        output = torch.matmul(output, y)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(2, 3, 4, 5)
y = torch.randn(2, 4, 5, 6)
