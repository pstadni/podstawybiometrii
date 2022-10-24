import cv2
import dlib
import math

# Wczytanie detectora
detector = dlib.get_frontal_face_detector()

# Ładowanie predictora
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Wczytanie obrazu
img = cv2.imread("face.jpg")

# Konwersja obrazu na grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

# Używanie detectora do znalezienia landmarks
faces = detector(gray)
for face in faces:
    x1 = face.left() # lewo
    y1 = face.top() # góra
    x2 = face.right() # prawo
    y2 = face.bottom() # dół

    # Tworzenie landmarku
    landmarks = predictor(image=gray, box=face)

    # Pętla przez wszystkie punkty
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

    xa = landmarks.part(36).x #prawe oko zew
    ya = landmarks.part(36).y
    praweokozew = [xa,ya]

    xb = landmarks.part(45).x  #lewe oko zew
    yb = landmarks.part(45).y
    leweokozew = [xb, yb]

    odl1= math.dist(praweokozew,leweokozew) #odległość między zewnętrznymi kącikami oczu
    print(odl1)

    xc = landmarks.part(39).x  # prawe oko wew
    yc = landmarks.part(39).y
    praweokowewn = [xc, yc]

    xd = landmarks.part(42).x  # lewe oko wew
    yd = landmarks.part(42).y
    leweokowewn = [xd, yd]

    odl2 = math.dist(praweokowewn, leweokowewn)  # odległość między wewnętrznymi kącikami oczu
    print(odl2)

    stosunek1 = abs(odl1/odl2)

    print('Stosunek odległości kącików oczu od siebie '+ str(stosunek1))


#52 do 8 to dolny brzeg ust do brody

    xe = landmarks.part(52).x  # dolny brzeg ust
    ye = landmarks.part(52).y
    dolust = [xe, ye]

    xf = landmarks.part(8).x  # broda
    yf = landmarks.part(8).y
    broda = [xf, yf]

    odl3 = math.dist(dolust, broda)  # odleglosc od ust do brody
    print(odl3)

# 6 i 10 szerokosc ust

    xg = landmarks.part(48).x  # prawy kacik ust
    yg = landmarks.part(48).y
    usta1 = [xg, yg]

    xh = landmarks.part(54).x  # lewy kacik ust
    yh = landmarks.part(54).y
    usta2 = [xh, yh]

    odl4 = math.dist(usta1, usta2)  # wybrana szerokosc brody
    print(odl4)

    stosunek2 = abs(odl3/odl4)
    print('Stosunek szerokości ust do odległości ust od brody '+ str(stosunek2))

cv2.imshow(winname="Face", mat=img)
cv2.waitKey(delay=0)
cv2.destroyAllWindows()
