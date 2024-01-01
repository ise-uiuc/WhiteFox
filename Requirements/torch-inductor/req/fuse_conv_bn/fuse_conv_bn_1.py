conv = torch.nn.ConvXd(...) # X can be 1, 2, or 3 representing the dimension
bn = torch.nn.BatchNormXd(...) # X should match with ConvXd
output = bn(conv(input_tensor))
