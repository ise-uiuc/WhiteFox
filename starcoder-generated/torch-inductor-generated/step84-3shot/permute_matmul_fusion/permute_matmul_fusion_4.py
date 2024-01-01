
t1 = torch.randn(3, 2, 1, 3, 1, 2) # A tensor with size [3, 2, 1, 3, 1, 2]
t2 = t1.permute(0, 4, 1, 3, 5, 2)
t3 = t2.reshape(t2.shape[0] * t2.shape[1], t2.shape[2] * t2.shape[3] * t2.shape[4] * t2.shape[5])
t4 = t2.reshape(3, 2, 1, -1)
model = Model(t3)
input = t1
loss = model(input)
