import win32gui
import win32ui
import win32con
import os
from PIL import Image
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import time



def screenshot_lower_right_quadrant(hwnd):
    # Obter o tamanho total da janela
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    # Coordenadas do corte baseadas em porcentagem da janela

    #definir porcentagens para capturar um quarto da tela inferior direita
    crop_left_pct = 0.75
    crop_top_pct = 0.75
    crop_right_pct = 1.1
    crop_bottom_pct = 1.2


    # Calcula as coordenadas de corte reais em pixels
    crop_left = int(width * crop_left_pct)
    crop_top = int(height * crop_top_pct)
    crop_right = int(width * crop_right_pct)
    crop_bottom = int(height * crop_bottom_pct)
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    # Ajustar coordenadas para o corte no momento da captura
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, crop_width, crop_height)
    cDC.SelectObject(dataBitMap)

    # Realiza a captura diretamente na área cortada
    cDC.BitBlt((0, 0), (crop_width, crop_height), dcObj, (crop_left, crop_top), win32con.SRCCOPY)

    # Converter o bitmap capturado em uma imagem PIL
    bmp_info = dataBitMap.GetInfo()
    bmp_str = dataBitMap.GetBitmapBits(True)
    screenshot = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_str, 'raw', 'BGRX', 0, 1)

    #save image for debug
    screenshot.save("screenshot.jpeg")

    # Limpar recursos
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return screenshot


def get_heart_crop_coordinates_from_screenshot(app_window_name):

    hwnd = win32gui.FindWindow(None, app_window_name)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width = right - left
    height = bottom - top

    quad_left = int(width * 0.75)
    quad_top = int(height * 0.75)


    screenshot = screenshot_lower_right_quadrant(hwnd)
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

    # Converte para o espaço de cor HSV para segmentar o vermelho
    hsv_image = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2HSV)

    # Define o intervalo de cor para detectar o vermelho do coração
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

    # Outra faixa para tons de vermelho, caso o coração seja um vermelho mais escuro
    lower_red_alt = np.array([170, 120, 70])
    upper_red_alt = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red_alt, upper_red_alt)

    # Combina ambas as máscaras
    mask = mask1 + mask2

    # Encontra os contornos da máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Pega o maior contorno, assumindo que é o coração
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Ajusta as coordenadas em relação à janela principal
        return x + quad_left - left, y + quad_top - top, w, h
    else:
        return None


def recognize(indice_imagem):

    image_path = f"./cropped_images/screenshot_cropped_{indice_imagem}.jpeg"

    image = Image.open(image_path)

    # Aumentar o contraste para melhorar a detecção
    enhancer = ImageEnhance.Contrast(image)
    
    enhanced_image = enhancer.enhance(2.0)

    # Aplicar filtro de nitidez para destacar os bordes do número
    sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)

    # Converter tudo que não é branco em preto
    # Pixels com valor 255 permanecem, o restante é convertido para 0
    binary_image = sharpened_image.point(lambda x: 255 if x == 255 else 0)
    
    new_size = (binary_image.width * 4, binary_image.height * 4)  # Dobrar o tamanho, por exemplo

    resized_image = binary_image.resize(new_size, Image.BICUBIC)

    # Usar pytesseract para extrair apenas números
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
    number = pytesseract.image_to_string(resized_image, config=custom_config)

    # Limpar e exibir o número extraído
    number = number.strip()  # Remove espaços extras
    return number

def background_screenshot_and_crop(hwnd, count):
    # Captura a tela completa da janela
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    crop_coords = get_heart_crop_coordinates_from_screenshot("cardioEmotion Home")
    x, y, w, h = crop_coords
    
    #Variaveis de ajuste para telas de diferentes proporções
    adjust_left = int(0.013 * right)
    adjust_right = int(0.033 * right)
    adjust_top = int(0.032 * bottom)
    adjust_bottom = int(0.066 * bottom)

    # Ajustar coordenadas de corte em relação à janela
    crop_left = left + x + adjust_left
    crop_top = top + y + adjust_top
    crop_right = crop_left + w - adjust_right
    crop_bottom = crop_top + h - adjust_bottom
    crop_width = crop_right - crop_left
    crop_height = crop_bottom - crop_top

    # Realizar a captura diretamente na área cortada
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, crop_width, crop_height)
    cDC.SelectObject(dataBitMap)

    # Captura a região definida pelo coração
    cDC.BitBlt((0, 0), (crop_width, crop_height), dcObj, (crop_left - left, crop_top - top), win32con.SRCCOPY)

    # Cria a pasta "cropped_images" se não existir
    output_dir = "cropped_images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva a imagem cortada diretamente na pasta "cropped_images"
    cropped_image_path = f"{output_dir}/screenshot_cropped_{count}.jpeg"
    dataBitMap.SaveBitmapFile(cDC, cropped_image_path)

    # Limpar recursos
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    print(f"Imagem cortada salva em: {cropped_image_path}")


def delete_image(i):
    os.remove(f"./cropped_images/screenshot_cropped_{i}.jpeg")



def extract_bpm():
    # Loop para capturar screenshots periodicamente com o corte adaptado ao coração
    # Definir o identificador da janela
    hwnd = win32gui.FindWindow(None, "cardioEmotion Home")
    index = 0
    crop_coords = get_heart_crop_coordinates_from_screenshot("cardioEmotion Home")
    if crop_coords:
        background_screenshot_and_crop(hwnd, index)
    else:
        print("Coração não encontrado na imagem.")
    bpm = recognize(index)
    os.remove(f"./cropped_images/screenshot_cropped_{0}.jpeg")

    return bpm
