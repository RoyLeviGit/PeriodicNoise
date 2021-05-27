import numpy as np
import matplotlib.image
import matplotlib.pyplot as plt


def get_image(file_name):
    image = matplotlib.image.imread(file_name)
    # reduce third dimension if it exists
    if len(image.shape) > 2:
        return np.dot(image[..., :1], [1])
    return image


def get_noisy_image(image, noise_a):
    noisy_image = image.copy()
    for k in range(int(2560 / 80)):
        for i in range(5):
            noisy_image_column = noisy_image[:, (k * 80) + (i * 16)]
            noisy_image_column += noise_a[i]
            # cycle values out of grayscale
            # noisy_image[:, (k * 80) + (i * 16)] = \
            #     noisy_image[:, (k * 80) + (i * 16)] % 256
            # clip values out of grayscale
            noisy_image_column[noisy_image_column > 255] = 255
            noisy_image_column[noisy_image_column < 0] = 0
    return noisy_image


def plot_image(image, title):
    plt.imshow(image, 'gray')
    plt.suptitle(title)
    plt.show()


def omega(n):
    return np.exp(-2j*np.pi/n)


def dft_entry(row, column, dft_size):
    return omega(dft_size) ** (row * column)


def dft(dft_size):
    dft_matrix = np.zeros((dft_size, dft_size), dtype=np.complex128)
    for row in range(dft_size):
        for column in range(dft_size):
            dft_matrix[row][column] = dft_entry(row, column, dft_size)
    return dft_matrix / np.sqrt(dft_size)


def periodic_noise():
    # Section a:
    image = get_image("road.jpeg")
    plot_image(image, "Original Image")
    noise_a = [int(x * 255 / 5) for x in range(-2, 4) if not x == 0]
    print(f"Noise to be added to columns of image: {noise_a}")
    noisy_image = get_noisy_image(image, noise_a)
    plot_image(noisy_image, "Noisy Image")

    # Section c:
    print(dft(4))

periodic_noise()

