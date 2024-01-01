
model = nn.Sequential(nn.Linear(2, 4), nn.Flatten(start_dim=1), torch.stack([x, x], dim=1), nn.Flatten(start_dim=2), nn.Linear(4, 2))
model(x)