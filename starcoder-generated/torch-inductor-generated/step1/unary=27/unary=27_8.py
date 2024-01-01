
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1, bias=False)
        self.conv.weight.data = torch.eye(8) * -0.0001
 
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp(input=v1, min=6380.7001953125)
        v3 = torch.clamp(input=v2, max=6380.13720703125)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x = torch.eye(3, 3)
print(m(x))

# Expected Output
# tensor([[-0.0001,  0.0000,  0.0000],
#         [ 0.0000, -0.0001,  0.0000],
#         [ 0.0000,  0.0000, -0.0001]], device='cuda:0', grad_fn=<ViewBackward>)
