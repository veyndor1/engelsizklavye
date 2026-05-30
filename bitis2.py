import cv2
import mediapipe as mp
import numpy as np
import pickle

model_dict = pickle.load(open("model.pickle", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

new_width = 1280
new_height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5:"F", 6:"G", 7:"H", 8:"I", 9:"J"}
captured_word = []
current_word = ""
previous_key = None 

x_pos, y_pos = 10, 50

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()



    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Kontrol 1: Modelin beklediği giriş boyutunu ve kullanılan veri boyutunu kontrol et
            print("Model Feature Count:", len(model.feature_importances_))
            print("Used Data Feature Count:", len(data_aux))

            if len(data_aux) != len(model.feature_importances_):
                continue
            prediction = model.predict([np.asarray(data_aux)])
            predicted_char = label_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("x"):
                current_word += predicted_char
                captured_word.append(predicted_char)
                previous_key = "x"
            elif key & 0xFF == ord(" "):
                current_word += " "
                captured_word.append(" ")

    cv2.putText(frame, "Kelime: " + current_word, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("İşaret dili", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

word = ''.join(captured_word)
print("Oluşturulan kelime: ", word)
