
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.cat([x1, x2])
        v2 = torch.cat([v1, x3])
        v3 = torch.cat([v2, x4])
        return v3[:, 0:size]
 
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 384) # Note the batch dimension is 1
x2 = torch.randn(1, 2, 448)
x3 = torch.randn(1, 3, 512)
x4 = torch.randn(1, 4, 576)
