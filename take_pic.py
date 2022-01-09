import os

import cv2
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from proj import Model,Danse


def take_pic(record = False):
    photo_list = []
    # define a video capture object
    vid = cv2.VideoCapture(1)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        #plt.imshow(frame)
        #plt.show()
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('p'):
            break

        if record:
            img = PIL.Image.fromarray(frame)
            img = np.array((img.resize((128,128))))
            photo_list.append(img)


    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

    img = PIL.Image.fromarray(frame)

    img = np.reshape(np.array((img.resize((128,128)).convert('L'))),(128*128,))

    return img,photo_list

def save_photo_list(path,photo_list,name = ''):
    os.chdir(f'{path}\\{name}')
    for id,photo in enumerate(photo_list):
        p = PIL.Image.fromarray(photo)
        p.save(f'{name}{id}.jpg')

rec = input('press enter to take a picture!')

if rec.lower() == 'rec':
    test,photo_list = take_pic(record = True)
    save_photo_list(path = 'D:\cyber\yb project\databases\photos',
                    photo_list = photo_list, name='Dvir')
else:
    test,photo_list = take_pic(record = False)


os.chdir(r'C:\Users\Dvir hamdi\PycharmProjects\cyberHW\yodbet project')
plt.imshow(np.reshape(test,(128,128)))
plt.show()

model = Model()
#model.train()
layer1 = Danse(n=128, inputs=16384,activtion_function = 'tanh')
layer2 = Danse(n=3, inputs=128,activtion_function = 'soft_max')

model.add(layer1)
model.add(layer2)

model.compile(loss='crossEntropy',lr = 0.00001)

pred_values = {0:'Dvir',1:'Dan',2:'Ron'}

model.load('model.p')
res = model.predict([test])
print('predictions:',res)
print(pred_values)
print(pred_values[np.argmax(res[0])])




