import cv2
import dlib
import math
from collections import Counter
import numpy as np


def generate_points(zdjecie):
    # Wczytanie detectora
    detector = dlib.get_frontal_face_detector()

    # Ładowanie predictora
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Wczytanie obrazu
    img = cv2.imread(zdjecie)

    # Konwersja obrazu na grayscale
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    # Używanie detectora do znalezienia landmarks
    faces = detector(gray)

    for face in faces:
        x1 = face.left()  # lewo
        y1 = face.top()  # góra
        x2 = face.right()  # prawo
        y2 = face.bottom()  # dół

        # Tworzenie landmarku
        landmarks = predictor(image=gray, box=face)

        # Pętla przez wszystkie punkty
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    xa = landmarks.part(36).x  # prawe oko zew
    ya = landmarks.part(36).y
    praweokozew = [xa, ya]

    xb = landmarks.part(45).x  # lewe oko zew
    yb = landmarks.part(45).y
    leweokozew = [xb, yb]

    odl1 = math.dist(praweokozew, leweokozew)  # odległość między zewnętrznymi kącikami oczu

    xc = landmarks.part(39).x  # prawe oko wew
    yc = landmarks.part(39).y
    praweokowewn = [xc, yc]

    xd = landmarks.part(42).x  # lewe oko wew
    yd = landmarks.part(42).y
    leweokowewn = [xd, yd]

    odl2 = math.dist(praweokowewn, leweokowewn)  # odległość między wewnętrznymi kącikami oczu
    stosunek1 = abs(odl1 / odl2)


    xe = landmarks.part(57).x  # dolny brzeg ust
    ye = landmarks.part(57).y
    dolust = [xe, ye]

    xf = landmarks.part(8).x  # broda
    yf = landmarks.part(8).y
    broda = [xf, yf]

    odl3 = math.dist(dolust, broda)  # odleglosc od ust do brody

    # 6 i 10 szerokosc ust

    xg = landmarks.part(48).x  # prawy kacik ust
    yg = landmarks.part(48).y
    usta1 = [xg, yg]

    xh = landmarks.part(54).x  # lewy kacik ust
    yh = landmarks.part(54).y
    usta2 = [xh, yh]

    odl4 = math.dist(usta1, usta2)  # wybrana szerokosc brody

    stosunek2 = abs(odl3 / odl4)
    # print('Stosunek 2: ' + str(stosunek2))

    xj = landmarks.part(0).x
    yj = landmarks.part(0).y
    z1 = [xj, yj]

    xl = landmarks.part(16).x
    yl = landmarks.part(16).y
    z2 = [xl, yl]

    odl5 = math.dist(z1, z2)

    xm = landmarks.part(27).x
    ym = landmarks.part(27).y
    z3 = [xm, ym]

    xn = landmarks.part(51).x
    yn = landmarks.part(51).y
    z4 = [xn, yn]

    odl6 = math.dist(z3, z4)

    stosunek3 = abs(odl5 / odl6)
    # print("Stosunek 3: " + str(stosunek3))

    xo = landmarks.part(17).x
    yo = landmarks.part(17).y
    z5 = [xo, yo]

    xp = landmarks.part(26).x
    yp = landmarks.part(26).y
    z6 = [xp, yp]

    odl7 = math.dist(z5, z6)

    xr = landmarks.part(21).x
    yr = landmarks.part(21).y
    z7 = [xr, yr]

    xs = landmarks.part(22).x
    ys = landmarks.part(22).y
    z8 = [xs, ys]

    odl8 = math.dist(z7, z8)

    stosunek4 = abs(odl7 / odl8)
    # print("Stosunek 4: " + str(stosunek4))

    # cv2.imshow(winname="Face", mat=img)
    # cv2.waitKey(delay=0)
    # # cv2.destroyAllWindows()

    arr = [stosunek1, stosunek2, stosunek3, stosunek4]
    arr_vector = np.array(arr)
    return arr_vector


def build_vector(iterable1, iterable2):
    counter1 = Counter(iterable1)
    counter2 = Counter(iterable2)
    all_items = set(counter1.keys()).union(set(counter2.keys()))
    vector1 = [counter1[k] for k in all_items]
    vector2 = [counter2[k] for k in all_items]
    return vector1, vector2


def cosim(v1, v2):
    dot_product = sum(n1 * n2 for n1, n2 in zip(v1, v2))
    magnitude1 = math.sqrt(sum(n ** 2 for n in v1))
    magnitude2 = math.sqrt(sum(n ** 2 for n in v2))
    return dot_product / (magnitude1 * magnitude2)


if __name__ == "__main__":

 # dane_dla_zdj1 = generate_points("/Users/paulinastadnik/PycharmProjects/pbproj/venv_p/1.jpg")

    a = input()
    wektor_cech = [2.19430116, 0.55347524, 2.42599661, 3.89185005]
    dane_dla_zdj2 = generate_points(str(a))
    wynik = (cosim(wektor_cech, dane_dla_zdj2))*10000 - 9900

    if wynik>95:
        print(wynik, "%")
        print("Twarz rozpoznana pozytywnie")
    else:
        print(wynik, "%")
        print("Twarz nie rozpoznana")
        print(cosim(wektor_cech, dane_dla_zdj2))

    # print(dane_dla_zdj1, dane_dla_zdj2)
