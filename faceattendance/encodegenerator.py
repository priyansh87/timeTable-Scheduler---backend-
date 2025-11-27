import cv2
import pickle
import face_recognition
import os

#importing students' images
folderPath = 'images'
print("Looking in folder:", os.path.abspath(folderPath))
imagePathList = os.listdir(folderPath)
print(imagePathList)
imgList = []
studentIdList = []

for imagePath in imagePathList:
    imgList.append(cv2.imread(os.path.join(folderPath, imagePath), cv2.IMREAD_COLOR))
    #print(imagePath)
    #print(os.path.splitext(imagePath)[0])
    studentIdList.append(os.path.splitext(imagePath)[0])
print(studentIdList)

def findEncoding(imgList):
    encodingList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)[0]
        encodingList.append(encodings)

    return encodingList

print("Encoding started")
encodingList = findEncoding(imgList)
encodingListwithIds = [encodingList,studentIdList]
print("Encoding finished")
print(encodingList)


file = open('EncodeFile.p', 'wb')
pickle.dump(encodingListwithIds, file)
file.close()
print("Encoding saved to 'EncodeFile.p'")