
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f_linear1 = torch.nn.Linear(32*32*3, 256)
        self.f_linear2 = torch.nn.Linear(256, 256)
 
    def forward(self, img):
        flat = img.view(-1, 32*32*3) # flatten the input
        f0 = self.f_linear1(flat)
        f1 = torch.relu(f0)
        f2 = self.f_linear2(f1)
        f3 = torch.relu(f2)
        # f_add is used for model size estimation
        f_add = f3
        x = torch.cat([f3], dim=1)
        # f_cat is used for model size estimation
        f_cat = x
        return f_add, f_cat

# Initializing the model
m = Model()

# Inputs to the model
img = torch.randn(1, 3, 32, 32)
