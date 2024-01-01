
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, x1):
        v1 = x1.view(-1) # Flatten the tensor
        v2 = v1.size(0)
        v3 = torch.zeros(v2, dtype=torch.float32, device=torch.device('cuda:0'))
        v4 = torch.sigmoid(v3)
        v5 = v1 * v4
        x2 = v5.view(1, -1)
        return x2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 200, dtype=torch.float32).cuda()
