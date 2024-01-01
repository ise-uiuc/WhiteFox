
model = torch.nn.Sequential(conv2d, torch.nn.BatchNorm2d(8))
# or equivalently
model = torch.nn.Sequential(conv2d, F.batch_norm)
conv2d.train(False)
