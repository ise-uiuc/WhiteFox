
class Model(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
 
    def forward(self, x):
        v1 = torch.full([x.size()[0],], 1, dtype=dtype, layout=torch.strided, device=device, pin_memory=False)
        v2 = v1.to(dtype)
        v3 = v2.to("cpu").to(dtype)
        v4 = torch.cumsum(v3, 1)
        return v4

# Initializing the model
m = Model("cpu", "float16")

# Inputs to the model
x = torch.randn(10, 20)
