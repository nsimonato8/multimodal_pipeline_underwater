import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

input_directory = r""
output_directory = r""


def create_directory(directory_path):
    """Crea una directory se non esiste già."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory creata: {directory_path}")


def simple_underwater_correction(image):
    """
    Applica una correzione base alle immagini subacquee.
    Questa è una versione semplificata che migliora i colori e il contrasto.
    """
    # Converti l'immagine in LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) al canale L
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Migliora i canali a e b (aumenta la saturazione)
    a = cv2.add(a, 10)  # Aumenta il rosso
    b = cv2.add(b, 10)  # Aumenta il giallo/blu

    # Unisci i canali
    updated_lab = cv2.merge((cl, a, b))

    # Converti nuovamente in BGR
    corrected = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

    # Applica un bilanciamento del bianco
    corrected = white_balance(corrected)

    # Migliora il contrasto
    corrected = improve_contrast(corrected, clahe)

    return corrected


def white_balance(img):
    """Applica un semplice bilanciamento del bianco utilizzando il metodo Gray World."""
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def improve_contrast(img, clahe):
    """Migliora il contrasto dell'immagine."""
    # Converti in HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Aumenta la saturazione
    s = cv2.multiply(s, 1.3)
    s = np.clip(s, 0, 255).astype(np.uint8)

    # Aumenta il valore (luminosità)
    v = clahe.apply(v)

    # Unisci i canali e converti nuovamente in BGR
    hsv = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return result


def process_images(input_dir, output_dir):
    """Elabora tutte le immagini nella directory di input e salva i risultati nella directory di output."""
    # Assicurati che la directory di output esista
    create_directory(output_dir)

    # Ottieni tutti i file nella directory di input
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f"*{ext}")))
        image_files.extend(list(Path(input_dir).glob(f"*{ext.upper()}")))

    print(f"Trovate {len(image_files)} immagini da elaborare.")

    # Elabora ogni immagine
    for img_path in tqdm(image_files, desc="Elaborazione immagini"):
        try:
            # Leggi l'immagine
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"Impossibile leggere l'immagine: {img_path}")
                continue

            # Applica la correzione
            corrected_img = simple_underwater_correction(img)

            # Salva l'immagine elaborata
            output_path = os.path.join(output_dir, os.path.basename(img_path))
            cv2.imwrite(output_path, corrected_img)

        except Exception as e:
            print(f"Errore nell'elaborazione dell'immagine {img_path}: {e}")


# if __name__ == "__main__":
#     # Definisci i percorsi di input e output

#     # Inizializza CLAHE come variabile globale per riutilizzo
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

#     # Processa le immagini
#     process_images(input_directory, output_directory)

#     print("Elaborazione completata!")
