import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt


# returns image from file
def get_image(file_name):
    image = matplotlib.image.imread(file_name)
    # reduce third dimension if it exists
    if len(image.shape) > 2:
        return np.dot(image[..., :1], [1])
    return image


# returns noisy image from original image and noise to be periodically added
def get_noisy_image(original_image, noise_a):
    noisy_image = original_image.copy()
    for k in range(int(2560 / 80)):
        for i in range(5):
            noisy_image_column = noisy_image[:, (k * 80) + (i * 16)]
            noisy_image_column += noise_a[i]

            # clip values out of grayscale
            noisy_image_column[noisy_image_column > 255] = 255
            noisy_image_column[noisy_image_column < 0] = 0
    return noisy_image


# plots given image
def plot_image(image, title):
    plt.imshow(image, 'gray')
    plt.suptitle(title)
    plt.show()


# omega as defined in class. used to calculate DFT entry
def omega(n):
    return np.exp(-2j*np.pi/n)


# returns DFT entry
def dft_entry(row, column, dft_size):
    return omega(dft_size) ** (row * column)


# returns DFT of given size. beware! O(n^2)
def dft(dft_size):
    dft_matrix = np.zeros((dft_size, dft_size), dtype=np.complex128)
    for row in range(dft_size):
        for column in range(dft_size):
            dft_matrix[row][column] = dft_entry(row, column, dft_size)
    return dft_matrix / np.sqrt(dft_size)


# returns notch filter with 0 for entries in the interference that are none zero and 1 otherwise
def get_notch_filter(interference_in_dft_domain):
    filter_indices = np.array([x for x in range(2560)
                               if abs(interference_in_dft_domain[x]) > 1e-8])
    notch_filter = np.ones(2560)
    notch_filter[filter_indices] = 0
    # as requested in section c (ii):
    notch_filter[0] = 1

    return notch_filter


# plot interference in the DFT domain and returns corresponding notch filter
def plot_interference_and_get_filter(noise_a, monster_dft):
    interference = np.zeros(2560)
    indices = np.array([x for x in range(2560) if x % 80 == 0])
    for i in range(5):
        interference[indices] = noise_a[i]
        indices += 16
    interference_in_dft_domain = (monster_dft @ interference)

    plt.plot(interference_in_dft_domain.real)
    plt.suptitle("Interference (real) in DFT domain")
    plt.show()

    plt.plot(interference_in_dft_domain.imag)
    plt.suptitle("Interference (imaginary) in DFT domain")
    plt.show()

    return get_notch_filter(interference_in_dft_domain)


# restore image using notch filter
def get_restored_image(noisy_image, monster_dft, monster_dft_conjugate, notch_filter):
    restored_image = noisy_image.copy()
    for row in range(2560):
        row_in_dft_domain = (monster_dft @ noisy_image[row, :])
        row_in_dft_domain = row_in_dft_domain * notch_filter
        # imaginary part should zero except for numerical errors
        restored_row = (monster_dft_conjugate @ row_in_dft_domain).real

        # clip values out of grayscale
        restored_row[restored_row > 255] = 255
        restored_row[restored_row < 0] = 0
        restored_image[row, :] = restored_row
    return restored_image


# calculates MSE between two images
def mse(original_image, other_image):
    mse_sum = 0
    for row in range(2560):
        for column in range(2560):
            mse_sum += (original_image[row][column] - other_image[row][column]) ** 2
    return mse_sum / (2560 ** 2)


# compares the MSE of the noisy image and the restored image with the original image
def compare_mse(original_image, noisy_image, restored_image):
    print(f"Noisy image MSE: {mse(original_image, noisy_image)}")
    print(f"Restored image MSE: {mse(original_image, restored_image)}")


def periodic_noise():
    # Section a:
    original_image = get_image("road.jpeg")
    plot_image(original_image, "Original Image")
    noise_a = [int(x * 255 / 5) for x in range(-2, 4) if not x == 0]
    print(f"Noise to be added to columns of image: {noise_a}")
    noisy_image = get_noisy_image(original_image, noise_a)
    plot_image(noisy_image, "Noisy Image")

    # Section b:
    monster_dft = dft(2560)
    monster_dft_conjugate = monster_dft.conjugate()
    notch_filter = plot_interference_and_get_filter(noise_a, monster_dft)

    # Section c:
    restored_image = get_restored_image(noisy_image, monster_dft, monster_dft_conjugate, notch_filter)
    plot_image(restored_image, "Restored Image")

    compare_mse(original_image, noisy_image, restored_image)


periodic_noise()
