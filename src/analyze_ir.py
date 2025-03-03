import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Load the calibrated infrared image
calibrated_image = cv2.imread("calibrated_ir_image.jpg", cv2.IMREAD_GRAYSCALE)

# Analyze pixel intensity distribution
def analyze_pixel_distribution(image):
    plt.figure(figsize=(8,5))
    plt.hist(image.ravel(), bins=50, color='red', alpha=0.7)
    plt.title("Pixel Intensity Distribution")
    plt.xlabel("Pixel Intensity (Temperature Approximation)")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# Detect anomalies (hotspots, cold spots, and dead pixels)
def detect_anomalies(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)

    # Define thresholds
    hotspot_threshold = mean_intensity + (2 * std_intensity)
    coldspot_threshold = mean_intensity - (2 * std_intensity)

    # Identify hotspot and coldspot pixels
    hotspots = (image > hotspot_threshold).astype(np.uint8) * 255
    coldspots = (image < coldspot_threshold).astype(np.uint8) * 255

    return hotspots, coldspots

# Generate heatmap of pixel intensity
def generate_heatmap(image):
    plt.figure(figsize=(8,6))
    sns.heatmap(image, cmap="inferno", xticklabels=False, yticklabels=False)
    plt.title("Infrared Sensor Heatmap")
    plt.show()

# Run analysis functions
analyze_pixel_distribution(calibrated_image)

hotspots, coldspots = detect_anomalies(calibrated_image)

# Display Hotspots & Coldspots
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(hotspots, cmap="hot")
plt.title("Hotspots Detected")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(coldspots, cmap="cool")
plt.title("Coldspots Detected")
plt.axis("off")

plt.show()

# Generate heatmap for visualization
generate_heatmap(calibrated_image)

# Save anomaly images
cv2.imwrite("hotspots_detected.jpg", hotspots)
cv2.imwrite("coldspots_detected.jpg", coldspots)
print("Hotspot & Coldspot images saved successfully.")
