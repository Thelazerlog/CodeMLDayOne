import cv2
import numpy as np

def verifier_qualite_image(image_path: str, seuil_laplacien: float = 100.0) -> dict:
    image = cv2.imread(image_path)
    if image is None:
        return {"accepter": False, "raison": "Image introuvable"}
        
    hauteur = 1000
    ratio = hauteur / image.shape[0]
    dim = (int(image.shape[1] * ratio), hauteur)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    accepter = variance > seuil_laplacien
    return {
        "accepter": accepter, 
        "score_nettete": variance,
        "raison": "Flou détecté, veuillez vous rapprocher et stabiliser" if not accepter else "OK"
    }

def aligner_sur_gabarit(image_path: str, template_path: str, max_features: int = 5000):
    img1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Photo prise par la sage-femme
    img2 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # Gabarit vierge
    
    sift = cv2.SIFT_create(max_features)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
            
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = img2.shape
        img_aligned = cv2.warpPerspective(cv2.imread(image_path), M, (w, h))
        return img_aligned, M
    else:
        raise ValueError("Pas assez de points de correspondance pour l'alignement.")