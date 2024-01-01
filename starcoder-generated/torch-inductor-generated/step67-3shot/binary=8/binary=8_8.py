
model_1 = torch.nn.ModuleList()
model_1.append(torch.nn.Hardsigmoid())
model_1.append(torch.nn.Hardsigmoid())
model_1.append(torch.nn.Hardsigmoid())
# Inputs to the model. Could also be inputs to the first layer of the model (e.g. model_1[0])
inputs_1 = torch.randn(1, 3, 224, 224)
