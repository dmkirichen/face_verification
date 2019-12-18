import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

import warnings
warnings.filterwarnings("ignore")

PATH_TO_IMGS = "./LabeledFacesInTheWild/"


def get_face(filename, required_size=(224, 224)):
    """
    Extracts face from the image.
    """

    pixels = plt.imread(filename)  # Getting array represantation of the image.
    detector = MTCNN()  # Using convnet for finding faces on the image.

    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']  # Getting coordinates of the first appeared face.
    face = pixels[y1: y1 + height, x1: x1 + width]  # Extracting face-array from original image.

    image = Image.fromarray(face)
    image = image.resize(required_size)

    return np.asarray(image)


def get_embeddings(filenames, path_to_dir='./images/'):
    """
    Get embeddings of the faces.
    """

    mod_filenames = [path_to_dir + fn for fn in filenames]

    faces = [get_face(fn) for fn in mod_filenames]  # Extracting face-arrays from images.
    samples = np.asarray(faces, 'float32')

    samples = preprocess_input(samples, version=2)

    # include_top = False  -->  model without classifier.
    # pooling = 'avg'  -->  h x w x d layer into 1 x 1 x d
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

    yhat = model.predict(samples)  # Getting vector-embedding of the face.
    return yhat


def matches(embedding1, embedding2, threshold=0.6):
    """
    Finds out, if the embedding corresponds to the same person.
    """

    # Calculates cosine distance between vectors.
    score = cosine(embedding1, embedding2)

    return (score <= threshold), score


def verification(fname1, fname2, threshold=0.6, path_to_dir='./images/', show_images=True):
    """
    Shows if given two images represent same person.
    """

    embed1, embed2 = get_embeddings((fname1, fname2), path_to_dir=path_to_dir)

    match, score = matches(embed1, embed2, threshold=threshold)

    if show_images:
        mod_filename1, mod_filename2 = path_to_dir + fname1, path_to_dir + fname2
        faces = np.concatenate((get_face(mod_filename1), get_face(mod_filename2)), axis=1)

        plt.figure()
        plt.imshow(faces)
        plt.show()

    if match:
        print("These pictures represent the same person. Cosine distance is {:.3f} (<{:.3f})".format(score, threshold))
    else:
        print("These pictures depict different people. Cosine distance is {:.3f} (>{:.3f})".format(score, threshold))


def verification_gui(filename1, filename2, threshold=0.6, path_to_dir=''):
    """
    Shows if given two images represent same person in gui
    """

    embed1, embed2 = get_embeddings((filename1, filename2), path_to_dir=path_to_dir)

    match, score = matches(embed1, embed2, threshold=threshold)

    return match, score


if __name__ == "__main__":
    
    # Showing examples of using the system
    filenames = ["Johnny_Depp/Johnny_Depp_0001.jpg",
                 "Johnny_Depp/Johnny_Depp_0002.jpg",
                 "Johnny_Depp/Johnny_Depp_0003.jpg",
                 "George_W_Bush/George_W_Bush_0001.jpg",
                 "Zico/Zico_0001.jpg"]

    fnames = ["Zorenko_Victoria_0001.jpg", "Zorenko_Victoria_0002.jpg", "Random_Person_1_0001.jpg"]

    verification(fnames[0], fnames[1])
    print("^ Should be positive. ^")

    verification(fnames[0], fnames[2])
    print("^ Should be negative. ^")
    exit(0)

    print('Positive Tests')
    verification(filenames[0], filenames[1], path_to_dir=PATH_TO_IMGS)
    verification(filenames[0], filenames[2], path_to_dir=PATH_TO_IMGS)
    print()
    print('Negative Tests')
    verification(filenames[0], filenames[3], path_to_dir=PATH_TO_IMGS)
    verification(filenames[0], filenames[4], path_to_dir=PATH_TO_IMGS)
