
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.zeros_like(v1).uniform_()
        __o = torch.nn.functional.relu(torch.matmul(v1, v1) + v2)  # 3-line model 
        return __o
    
# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
o = m(x)

print ("Shape of output: ", o.shape)

if args.model_type_int == 0:
    print(o[0][0][0].cpu().item())
else:
    print(o.cpu().item())

