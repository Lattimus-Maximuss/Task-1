import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from google.colab import files
import io

# 1. Define a simple model
class Color2Gray(nn.Module):
    def __init__(self):  
        super(Color2Gray, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# 2. Upload an image in Colab
uploaded = files.upload()
filename = list(uploaded.keys())[0]
image = Image.open(io.BytesIO(uploaded[filename])).convert("RGB")

# 3. Preprocess image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
color_img = transform(image).unsqueeze(0)  # shape (1,3,128,128)


# 4. Initialize model, loss, optimizer
model = Color2Gray()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 5. Create grayscale target
gray_target = transforms.Grayscale()(image)
gray_target = transform(gray_target).unsqueeze(0)  # shape (1,1,128,128)


# 6. Training loop
for epoch in range(20):
    optimizer.zero_grad()
    output = model(color_img)
    loss = criterion(output, gray_target)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/20], Loss: {loss.item():.4f}")


# 7. Show results
with torch.no_grad():
    predicted_gray = model(color_img).squeeze().numpy()

plt.figure(figsize=(8,4))

plt.subplot(1, 2, 1)
plt.title("Original Color")
plt.imshow(color_img.squeeze().permute(1, 2, 0))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Grayscale")
plt.imshow(predicted_gray, cmap="gray")
plt.axis("off")

plt.show()
