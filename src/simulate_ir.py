import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generate a synthetic infrared image (random heat signatures)
def generate_infrared_image(width=640, height=480):
    # Simulate pixel intensity (temperature-like data) using a random heat map
    infrared_image = np.random.uniform(50, 255, (height, width)).astype(np.uint8)

    # Apply a colormap to make it visually similar to an infrared camera output
    infrared_colored = cv2.applyColorMap(infrared_image, cv2.COLORMAP_INFERNO)

    return infrared_colored

# Generate and display the infrared image
infrared_image = generate_infrared_image()

plt.figure(figsize=(8,6))
plt.imshow(cv2.cvtColor(infrared_image, cv2.COLOR_BGR2RGB))
plt.title("Simulated Infrared Image")
plt.axis("off")
plt.show()

# Save the image for future processing
cv2.imwrite("simulated_ir_image.jpg", infrared_image)
print("Infrared image saved as 'simulated_ir_image.jpg'")
