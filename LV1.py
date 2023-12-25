import numpy as np
import cv2
import matplotlib.pyplot as plt

#2d fft transforamcija nad slikom, dobija se frekvencijska reprezentacija slike
def fft(slika):
    slika_fft = np.fft.fft2(slika)
    slika_fft = np.fft.fftshift(slika_fft)
    return slika_fft

#kompleksna reprezentacija u frekvencijskom domenu
def inverse_fft(magnitude_log, complex_module_1):
    slika_fft = complex_module_1 * np.exp(magnitude_log)
    slika_filtrirana = np.abs(np.fft.ifft2(slika_fft))
    return slika_filtrirana

#nalazi sva 4 piksela i postavlja njihovu vrednost na 0
def noise_removal(slika): #uklanjanje suma
    slika_fft = fft(slika)
    slika_fft_mag = np.abs(slika_fft)
    slika_mag_1 = slika_fft/slika_fft_mag #skaliranje amplitude
    slika_fft_log = np.log(slika_fft_mag) # logaritam magnituda
    plt.imshow(np.abs(slika_fft_log))
    plt.show()

#redosled je y,x
    tacke = ((246, 246),(246, 306),(266, 206),(266, 266))
    amplituda = 0 #za tacke koje izazivaju sum amplituda se postavlja na 0

    for tacka in tacke:
        slika_fft_log[tacka[0], tacka[1]] = amplituda

    plt.imshow(np.abs(slika_fft_log)) #magnituda posle
    plt.show()

    slika_filtrirana = inverse_fft(slika_fft_log, slika_mag_1)
    return slika_filtrirana

if __name__ == '__main__':
    slika = cv2.imread("./slika_1.png")
    slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY) #convert u crno-belo
    plt.imshow(slika, cmap='gray')
    plt.show()

    slika_filtrirana = noise_removal(slika)

    plt.imshow(slika_filtrirana,cmap='gray')
    plt.show()
