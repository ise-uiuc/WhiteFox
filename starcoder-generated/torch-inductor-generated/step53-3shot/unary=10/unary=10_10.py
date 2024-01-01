
class Model(torch.nn.Module):
   def forward(self, x1):
       y1 = torch.clamp_min(torch.clamp_max(torch.nn.Linear(65056, 4096)(x1), 6.), 0.)
       y2 = torch.clamp_min(torch.clamp_max(torch.nn.Linear(4096, 4096)(y1), 6.), 0.)
       y3 = torch.clamp_min(torch.clamp_max(torch.nn.Linear(4096, 4096)(y2), 6.), 0.)
       y4 = torch.clamp_min(torch.clamp_max(torch.nn.Linear(4096, 4096)(y3), 6.), 0.)
       y5 = torch.clamp_min(torch.clamp_max(torch.nn.Linear(4096, 512)(y4), 6.), 0.)
       return y5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 65056)
