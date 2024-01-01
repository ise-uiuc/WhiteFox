
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1):
        v1 = x1.mean([1], keepdim=True)
        v2 = v1 * torch.tensor([[255., 255., 255.]], dtype=torch.float32)
        return v2
    
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.rand(1, 3, 256, 256) * 256
