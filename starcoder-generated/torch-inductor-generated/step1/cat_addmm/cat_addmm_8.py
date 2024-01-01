
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 3, 3, 3))
    
    def forward(self, x):
        v1 = torch.sigmoid(self.weight)
        v3 = x * v1
        v4 = F.conv2d(v3, v3)
        v5 = v4.view(1, 16, 16)
        return F.cat([(v4 + v1) * 0.25 + torch.tanh(v4) + v1, v5], dim=1) 
    
@RegisterModel("torch_op_test_model")
def create_torch_op_test_model(device="cpu"):
    