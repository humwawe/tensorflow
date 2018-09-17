import cv2

if __name__ == '__main__':
    face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread('face.jpg')
    cv2.imshow('src', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_xml.detectMultiScale(gray, 1.3, 5)
    print('face=', len(faces))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_face = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        cv2.imwrite("dst.jpg", roi_color)
        eyes = eye_xml.detectMultiScale(roi_face)
        print('eye=', len(eyes))
        for (e_x, e_y, e_w, e_h) in eyes:
            cv2.rectangle(roi_color, (e_x, e_y), (e_x + e_w, e_y + e_h), (0, 255, 0), 2)
    cv2.imshow('dst', img)
    cv2.waitKey(0)
