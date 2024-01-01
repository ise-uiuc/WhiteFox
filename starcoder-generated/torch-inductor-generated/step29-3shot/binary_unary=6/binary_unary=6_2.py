
class Model(torch.nn.Module):
    def forward(self, x2):
        v1 = linear(x2)
        v2 = v1 - other
        v3 = relu(v2)
        return v3
    
# Initializing the model
m = Model()
