import requests

url = "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/sAHGrMJHZLtkbg" 
output = "Ilya_model.rar"

# Яндекс.Диск часто требует подтверждения, поэтому лучше использовать прямую ссылку
# Попробуйте заменить URL на:
# url = "https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/XXXXX"

try:
    response = requests.get(url, allow_redirects=True, stream=False)
    response.raise_for_status()  # Проверка на ошибки
    
    total_size = int(response.headers.get('content-length', 0))
    print(f"Загружаем файл {output} ({total_size/1024/1024:.2f} MB)")
    
    with open(output, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Файл успешно скачан!")
except Exception as e:
    print(f"Ошибка при загрузке: {e}")
