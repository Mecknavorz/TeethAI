from unet import create_u_net

model = create_u_net(1024, 1024, 3)
model.summary()