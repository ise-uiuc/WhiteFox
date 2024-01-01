
model = torch.nn.Sequential(OrderedDict([
  ('conv1', torch.nn.Conv2d(3, 64, (7, 7), stride=(1, 1), padding=(3, 3))),
  ('conv2', torch.nn.Conv2d(64, 16, (1, 1), stride=(1, 1), padding=(0, 0))),
  ('conv3', torch.nn.Conv2d(16, 4, (1, 1), stride=(1, 1), padding=(0, 0))),
  ('conv4', torch.nn.Conv2d(4, 1, (1, 1), stride=(1, 1), padding=(0, 0))),
  ('ReLU', torch.nn.ReLU())]))
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
