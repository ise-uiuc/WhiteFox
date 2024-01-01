
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
 
    def test_layer(self):
        t1 = torch.randn(1, 64, 10, 5)
        t2 = torch.randn(64, 64)
        t3 = torch.matmul(t1, t2)
        t4 = self.fc(t3)
        return torch.cat([t1], dim=1)
 
# Initializing the model
m = Model()

# Outputs of the model
