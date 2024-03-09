from PIL import Image
from waifu_scorer import WaifuScorer

if __name__ == '__main__':
    predictor = WaifuScorer()
    image = Image.open("/path/to/your/image.jpg")
    score = predictor.predict(image)[0]
    print(score)
