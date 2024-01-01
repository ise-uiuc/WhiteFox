
class Model(torch.nn.Module):
    def __init__(self, num_head):
        super().__init__()
        self.num_head = num_head
 
    def forward(self, x1):
        x2 = torch.matmul(x1, x1)
        x3 = torch.nn.functional.dropout(x3, p=0.1)
        x5 = x3 * x2
        return x5

# Initializing the model
m = Model(num_head=56)

# Inputs to the model
x1 = torch.randn(1, 1024, 2048)
