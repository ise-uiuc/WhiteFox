
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5)
        self.conv2 = torch.nn.Conv2d(3, 8, 5)
 
    def forward(self, x1, x2):
        q = self.conv1(x1)
        k = self.conv2(x2)
        b, c, h, w = q.size()
        q = q.transpose(-3, -2).reshape(b, c, -1).transpose(-1, -2)
        k = k.transpose(-3, -2).reshape(b, c, -1).transpose(-1, -2)
        dk = float(c) ** -0.5
        attn = (q @ k.transpose(-1, -2)) * dk
        softmax_attn = attn.softmax(dim=-1)
        dropout_attn = torch.nn.functional.dropout(softmax_attn, p=0.1)
        output = (dropout_attn @ v).transpose(-2, -1).reshape(b, c, h, w)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
x2 = torch.randn(1, 3, 224, 224)
