import os
import pickle
import time
from datetime import datetime

from matplotlib import pyplot as plt
from pytorch_msssim import ssim
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from builder.EncoderForecasterBase import EncoderForecasterBase
from builder.TensorBuilder import multioutput_tensor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Calculating on device: {device}')
batch_size = 10
epochs = 1000
learning_rate = 1e-3

data_freq = 7
sea_name = 'barents'
loss_name = 'l1'

DIMS = None
x_virg = []
temp_ar = []
for file in os.listdir(f'matrices/{sea_name}_sea_osisaf'):
    date = datetime.strptime(file, f'osi_{sea_name}_%Y%m%d.npy')
    if date.year < 2016:
        array = np.load(f'matrices/{sea_name}_sea_osisaf/{file}')
        DIMS = array.shape
        temp_ar.append(array)
        if len(temp_ar) == data_freq:
            temp_ar = np.array(temp_ar)
            temp_ar = temp_ar[-1]
            x_virg.append(temp_ar)
            temp_ar = []
    else:
        break

x_virg = np.array(x_virg)

pre_history_size = 104
forecast_size = 52

dataset = multioutput_tensor(pre_history_size, forecast_size, x_virg)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
print('Loader created')

encoder = EncoderForecasterBase()
encoder.init_encoder(input_size=[DIMS[0], DIMS[1]],
                     n_layers=5,
                     in_channels=pre_history_size,
                     out_channels=forecast_size)
encoder.to(device)
print(encoder)

optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
criterion = nn.L1Loss()

losses = []
start = time.time()
for epoch in range(epochs):
    loss = 0
    for train_features, test_features in dataloader:
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        optimizer.zero_grad()
        outputs = encoder(train_features)
        if loss_name == 'l1':
            l1_loss = criterion(outputs, test_features)
            train_loss = l1_loss
        if loss_name == 'ssim':
            ssim_loss = 1 - ssim(outputs, test_features, data_range=1, size_average=True)
            train_loss = ssim_loss
        train_loss.backward()
        optimizer.step()
        loss += train_loss.item()

    loss = loss / len(dataloader)
    losses.append(loss)
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

end = time.time() - start
print(f'Runtime seconds: {end}')
torch.save(encoder.state_dict(), f"single_models/{sea_name}_104_52_{loss_name}(1990-2015).pt")
plt.plot(np.arange(len(losses)), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()