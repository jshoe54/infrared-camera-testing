import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the simulated infrared image
infrared_image = cv2.imread("simulated_ir_image.jpg", cv2.IMREAD_GRAYSCALE)

# Flat-Field Correction (FFC)
def apply_flat_field_correction(image):
    # Simulate a calibration frame (average of multiple IR frames in real-world)
    calibration_frame = cv2.GaussianBlur(image, (15, 15), 0)
    
    # Normalize image using calibration frame
    corrected_image = cv2.subtract(image, calibration_frame)
    corrected_image = cv2.normalize(corrected_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return corrected_image.astype(np.uint8)

# Noise Reduction (Denoising using Gaussian Blur)
def reduce_noise(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Apply calibration steps
calibrated_image = apply_flat_field_correction(infrared_image)
denoised_image = reduce_noise(calibrated_image)

# Apply colormap to visualize correction
calibrated_colored = cv2.applyColorMap(denoised_image, cv2.COLORMAP_INFERNO)

# Display the original and corrected images
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.applyColorMap(infrared_image, cv2.COLORMAP_INFERNO))
plt.title("Original Infrared Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(calibrated_colored, cv2.COLOR_BGR2RGB))
plt.title("Calibrated & Denoised Infrared Image")
plt.axis("off")

plt.show()

# Save the processed image
cv2.imwrite("calibrated_ir_image.jpg", calibrated_colored)
print("Calibrated infrared image saved as 'calibrated_ir_image.jpg'")
