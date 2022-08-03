import cv2, time, pandas
from datetime import datetime


video = cv2.VideoCapture(0)
first_frame = None
status_list =[None, None]
times = []

df = pandas.DataFrame(columns=["Start", "End"])

while True:
    check, frame = video.read()
    status = 0 #denotes that there's no motion yet
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray) #works out difference between the first frame and the current frame

    thresh_frame= cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] #if the diff between pixels in a place in the first frame and the current frame > 30, color those pixels white

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Finds the contours(outlines) of all the distinct objects between current frame and first frame

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1 #changed to 1 because now there is motion
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0, 255, 0), 3)

    status_list.append(status)


    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Capturing", gray)
    cv2.imshow("Deta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)
    

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
           times.append(datetime.now()) 
        break

print(status_list)
print(times)

for i in range(0,len(times), 2):
    df=df.append({"Start": times[i], "End":times[i+1]}, ignore_index=True)


df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows
