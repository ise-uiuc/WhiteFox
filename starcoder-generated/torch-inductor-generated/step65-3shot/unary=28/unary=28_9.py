
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = self.layer1(x)
        v2 = torch.clamp_max(v1, max=18420646400000000.0)
        v3 = torch.clamp_min(v2, min=0.0)
        return v3
    
# Initializing the model
m = Model()

# The model expects input images of resolution 64x64 in RGB format
