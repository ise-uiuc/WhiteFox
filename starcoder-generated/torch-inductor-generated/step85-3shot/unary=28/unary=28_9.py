
class Model(torch.nn.Module):
    def forward(self,x):
        v1 = linear(x)
        return torch.clamp_min(torch.clamp_max(v1,max_value=1),min_value=0)

# Initializing the model
m = Model()
 
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
