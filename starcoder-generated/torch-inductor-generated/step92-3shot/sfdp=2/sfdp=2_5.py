
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transpose0 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.transpose1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
 
    def forward(self, x1, x2):
        v1 = self.transpose0(x1)
        v2 = self.transpose1(x2)
        q = torch.matmul(v1, v2.transpose(-2, -1))
        qk = q / math.sqrt(self.model_dim)
        softmax_qk = qk.softmax(dim=-1)
        output = softmax_qk.matmul(v2)
        return output
 
# Initializing the model
m = Model(model_dim=256)

# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
x2 = torch.randn(1, 8, 16, 16)
