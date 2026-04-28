import cv2

print("Пошук підключених камер у Windows...")
available_cameras = []

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ УСПІХ: Камера знайдена на індексі [{i}]")
            available_cameras.append(i)
            
            cv2.imshow(f"Camera Index {i}", frame)
            cv2.waitKey(1500)
            cv2.destroyAllWindows()
        else:
            print(f"⚠️ Індекс [{i}] відкрився, але відео немає (можливо, зайнята іншою програмою).")
        cap.release()
    else:
        print(f"❌ Індекс [{i}] — порожньо.")

print(f"\nПідсумок: Вам потрібно використовувати індекси: {available_cameras}")
