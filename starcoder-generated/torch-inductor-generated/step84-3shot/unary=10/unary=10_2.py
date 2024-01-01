: Scaled and shifted ReLU6 activation function
class ScaledReLU6(torch.nn.Module):
  def forward(self, x):
    return torch.clamp_min(torch.clamp_max((torch.nn.functional.relu6(x + 3) * 6) / 6, 0), 6)

# Initializing the model
srelu6_m = ScaledReLU6()

# Inputs to the model
x2 = torch.randn(1, 4, 64, 64)
