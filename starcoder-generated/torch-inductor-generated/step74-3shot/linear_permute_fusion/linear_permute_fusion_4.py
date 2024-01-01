
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        x1_s = x1.size()
        x2_s = x2.size()
        x3_s = x3.size()
        x1_e = torch.reshape(x1, (-1,))
        x2_e = torch.reshape(x2, (-1,))
        x3_e = torch.reshape(x3, (-1,))
        x_e = torch.cat((x1_e, x2_e, x3_e), 0)  
        x_s = x_e.size()
        x = torch.reshape(x_e, x_s[0], x_s[1], int(x_s[2]/3), 3)
        v1 = torch.nn.functional.relu(x)
        v2 = torch.tanh(v1)
        v3 = torch.nn.functional.softmax(v2)
        return v2
# Inputs to the model
x1 = torch.randn(2, 4, 2, device='cpu')
x2 = torch.randn(2, 4, 2, device='cpu')
x3 = torch.randn(2, 4, 2, device='cpu')
