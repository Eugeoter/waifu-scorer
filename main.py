from PIL import Image
from modules import WaifuFilter

if __name__ == '__main__':
    predictor = WaifuFilter()
    image = Image.open("/path/to/your/image.jpg")
    score = predictor.predict(image)[0]
    print(score)
