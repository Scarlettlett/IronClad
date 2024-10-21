import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

class Embedding:

    def __init__(self, pretrained='casia-webface', device='cpu'):
        # Initialize the FaceNet model and set it to evaluation mode
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)

    def encode(self, image):
        # Move the image to the same device as the model
        image = image.to(self.device)

        # Get the embedding for the image
        with torch.no_grad():
            embedding = self.model(image)

        return embedding.squeeze().cpu().numpy()



if __name__ == "__main__":
    from PIL import Image
    from preprocess import Preprocessing

    image_size = 160
    preprocessing = Preprocessing(image_size=image_size)
    image_path = "simclr_resources/probe/Alan_Ball/Alan_Ball_0002.jpg"
    probe = Image.open(image_path)
    probe = preprocessing.process(probe)

    model = Embedding()

    print(model.encode(probe).shape)
    print(model.encode(probe))