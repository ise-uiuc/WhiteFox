
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 3, 1, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        return v2

# Initializing the model
m = Model()

# Input to the model
x = torch.randn(1, 3, 64, 64)

# Randomly generating the minimum and maximum values
minimum = random.choice([0.9, 0.5, 0.1])
maximum = random.choice([1.0, 0.8, 0.6])

# Clamping the outputs of the layers to the minimum and maximum values
v1 = torch.clamp_min(v1, minimum)
v2 = torch.clamp_max(v2, maximum)

# Adding a dummy print statement to be able to save the model from memory
print(v2)

# Saving the model
torch.save(m, Path('12_script_module.pt'))
