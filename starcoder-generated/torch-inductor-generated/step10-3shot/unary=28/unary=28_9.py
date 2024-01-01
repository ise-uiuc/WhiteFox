
class Model(module.Module):
    def __init__(self,min_value,max_value):
        super().__init__()
        self.linear=module.Linear(in_features=128,out_features=729,bias=True)
        self.min_value=min_value
        self.max_value=max_value

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1,self.min_value)
        v3 = torch.clamp_max(v2,self.max_value)
        return v3

# Initializing the model
m = torch.nn.ReLU()
m = Model(-0.4692587411870063,0.7854830915257639)

# Inputs to the model
x1 = torch.randn(1,128)
