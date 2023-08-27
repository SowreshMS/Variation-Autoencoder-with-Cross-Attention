from datasets import load_dataset
import torch
from torch.cuda.amp import autocast, GradScaler
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import tiktoken
import cv2
from vae import VAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

!mkdir -p "/content/drive/My Drive/models"

def download_image(url, timeout=30):
  try:
      response = requests.get(url, timeout=timeout)
      response.raise_for_status()  # Check for any errors in the response
      image_data = BytesIO(response.content)
      
      image = Image.open(image_data)
      pixel_values = np.array(image)
      
      return pixel_values
  except requests.exceptions.RequestException as e:
      pass


def tokenize(x):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(x)  
    ids.append(enc.eot_token)  
    out = {'ids': ids, 'len': len(ids)}
    targets = torch.tensor(out['ids'])
    return targets

num_epochs = 5
batch_size = 1
learning_rate = 0.001
torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VAE().to(device)

dataset = load_dataset('laion/laion2B-en', split='train', streaming=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

gradient_accumulator = 32

for epoch in range(num_epochs):
  losses = []
  for i, batch in enumerate(dataset):
    
    images = batch['URL']
    text = batch['TEXT']

    try:
      images = download_image(images)
      images = cv2.resize(images, (128, 128))
      images = torch.tensor(images.reshape(1, 3, 128, 128)).to(device)
    except:
      continue

    images = images.to(torch.float32)  / 255.0

    text = tokenize(text).unsqueeze(0).to(device)

    optimizer.zero_grad()

    # with autocast():
    x_recon, mu, log_var = model(images, text)

    loss = kl_loss(x_recon, images, mu, log_var)
    loss /= gradient_accumulator

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # scaler.scale(loss).backward()
    loss.backward()

    losses.append(loss.item() * 32)

    if (i + 1) % gradient_accumulator == 0:
      # scaler.step(optimizer)
      # scaler.update()
      print(sum(losses) / len(losses))
      optimizer.step()
      

    if (i + 1) % 50 == 0:
      torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'batch_num': i, 'last_seen_data': batch}, f"/content")
      with open("/content/drive/My Drive/models/metrics.txt", 'w') as file:
        file.write(f'Epoch: {i} Loss: {sum(losses) / len(losses)}')
