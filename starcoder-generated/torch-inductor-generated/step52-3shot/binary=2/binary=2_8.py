
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Embedding(51,3)
    def forward(self, p1):
        v1 = self.layer(p1)
        v2 = v1 - 3.1415
        return v2
# Inputs to the model
p1 = torch.randint(3,51,(1,10))
