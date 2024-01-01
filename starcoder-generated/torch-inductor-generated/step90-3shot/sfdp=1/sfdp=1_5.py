
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x3, x6):
        mv1 = torch.matmul(x3, x6.transpose(-2, -1))
        mv2 = mv1.div(inv_scale_factor)
        mv3 = mv2.softmax(dim=-1)
        mv4 = torch.nn.functional.dropout(mv3, p=dropout_p)
        out = mv4.matmul(x6)
        return out

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(1, 32, 4, 4)
x6 = torch.randn(1, 4, 16, 16)
