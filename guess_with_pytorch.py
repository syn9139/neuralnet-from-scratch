import numpy as np
import matplotlib.pyplot as plt
import torch
from training_with_pytorch import *
data = np.load('weights_with_pytorch.npz')

# Import a random image
sample = np.random.randint(0, 10000)
with torch.no_grad():
    for i, p in enumerate(params):
        p.copy_(torch.from_numpy(data[f'arr_{i}']))
    logits = forward_pass(test_image[sample])
    output = torch.softmax(logits, dim=-1)

 
# Show actual image and prediction
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(test_image[sample].reshape(28, 28).numpy(), cmap='gray')
ax1.set_title(f"Label: {test_label[sample].item()}")
ax1.axis('off')
 
ax2.bar(range(10), output)
ax2.set_xticks(range(10))
ax2.set_title(f"Guess: {np.argmax(output)}")
 
plt.show()

# Calculate success rate
with torch.no_grad():
    predictions = torch.softmax(forward_pass(test_image), dim=-1).argmax(dim=1)
    success_rate = (predictions == test_label).float().mean().item() * 100
print(f"Success rate: {success_rate:.2f}%")
