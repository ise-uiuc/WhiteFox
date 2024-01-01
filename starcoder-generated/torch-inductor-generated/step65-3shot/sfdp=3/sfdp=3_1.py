
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2, x3):
        qk = torch.matmul(x1, x2.transpose(-2, -1))
        scale_factor = torch.sqrt(torch.tensor(x3.size(-1))).to(x3.device)
        softmax_qk = torch.nn.functional.softmax(qk / scale_factor, dim=-1)
        dropout_output = torch.nn.functional.dropout(softmax_qk, p=0.1)
        output = dropout_output.matmul(x3)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 8, 8)
