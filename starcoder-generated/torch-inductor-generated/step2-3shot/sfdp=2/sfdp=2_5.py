
class Model(torch.nn.Module):
    def __init__(self:
        super().__init__()
        
    def forward(self, __input1__, __input2__, __input3__):
        v1 = torch.matmul(__input1__, __input2__.transpose(-2, -1))
        v2 = v1.div(1)
        v3 = v2.softmax(dim=-1)
        v4 = torch.nn.functional.dropout(v3, p=0.3)
        v5 = v4.matmul(__input3__)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 100, 50).cuda()
x2 = torch.randn(20, 50, 800).cuda()
x3 = torch.randn(20, 800, 1024).cuda()
