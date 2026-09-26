import numpy as np
import matplotlib.pyplot as plt
from training import *

with np.load('weights.npz') as data:
    for name, p in zip(['W1', 'b1', 'W2', 'b2', 'W3', 'b3'], params):
        p[:] = data[name]

a0, z1, a1, z2, a2, z3, a3 = forward(test_img_data)
sample = np.random.randint(0, test_img_data.shape[1])

# Show actual image and prediction
output = a3[sample, :]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.imshow(test_img_data[:, sample].reshape(28, 28), cmap='gray')
ax1.set_title(f"Label: {test_label_data[sample]}")
ax1.axis('off')

ax2.bar(range(10), output)
ax2.set_xticks(range(10))
ax2.set_title(f"Guess: {np.argmax(output)}")

plt.show()

# Calculate success rate
predictions = np.argmax(a3, axis=1)
success_rate = np.mean(predictions == test_label_data) * 100
print(f"Success rate: {success_rate:.2f}%")