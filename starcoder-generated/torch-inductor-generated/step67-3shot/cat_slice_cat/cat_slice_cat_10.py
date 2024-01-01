
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cat = torch.cat
 
    def forward(self, x):
        v1, v2 = self.cat((x, x), dim=1)
        v1_slice = v1[:, 0:9223372036854775807]
        v1_slice_2 = v1_slice[:, 0:v1.size(2)]
        v2, _ = self.cat((v1_slice_2, v2), dim=1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 10, 1)
