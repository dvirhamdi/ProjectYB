import os

import cv2
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
from proj import Model,Danse
import HD

def take_pic(record = False):
    photo_list = []
    # define a video capture object
    vid = cv2.VideoCapture(0)

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

    #img = np.reshape(np.array((img.resize((128,128)).convert('L'))),(128*128,))
    #return  frame,1
    return img,photo_list

def save_photo_list(path,photo_list,name = ''):
    os.chdir(f'{path}\\{name}')
    for id,photo in enumerate(photo_list):
        p = PIL.Image.fromarray(photo)
        p.save(f'{name}{id}.jpg')


def load_image(path):
    img = np.array((PIL.Image.open(path)))
    img = HD.face_dedection(img)

    return img


rec = input('press enter to take a picture!')


if rec.lower() == 'rec':
    test,photo_list = take_pic(record = True)
    save_photo_list(path = 'D:\cyber\yb project\databases\photos',
                    photo_list = photo_list, name='Dvir')
elif rec.lower() == 'vid':
    test,photo_list = take_pic(record = True)
    photo_list = np.array(photo_list[::5])
    test = photo_list
    t = []
    for i in test:
        t.append(HD.face_dedection(i))
        print(t)
    test = np.copy(t)
else:
    test,photo_list = take_pic(record = False)
    test = np.array(test)
    test = HD.face_dedection(test)[0]
    print(test)
    test = np.reshape(test,(1,128,128))
    #test = cv2.resize(test, (128,128), interpolation = cv2.INTER_AREA)
    #test = np.array(test)
    #test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    #test = np.reshape(test,(1,128,128))

#print(test)


test = test / 255

print(test)
#print(test.shape)
#test = np.reshape(test,(1,128*128))
print(test.shape)
#test = load_image(r'D:\cyber\yb project\databases\photos\test\Dvir\Dvir924.jpg')
test = np.reshape(test,(test.shape[0],128*128))
model = Model()
#model.train()
layer1 = Danse(n=16384//2, inputs=16384,activtion_function = 'relu')
layer2 = Danse(n=256, inputs=16384//2,activtion_function = 'relu')
layer3 = Danse(n=4, inputs=256,activtion_function = 'soft_max')

model.add(layer1)
model.add(layer2)
model.add(layer3)



model.compile(loss='crossEntropy',lr = 0.00001)

pred_values = {0:'Dan',1:'Dvir',2:'Fefer',3:'Guy'}

model.load(r'C:\Users\Dvir hamdi\PycharmProjects\cyberHW\yodbet project\model.p')




os.chdir(r'C:\Users\Dvir hamdi\PycharmProjects\cyberHW\yodbet project')
plt.imshow(np.reshape(test[0],(128,128)))
plt.show()

res = model.predict(test)
print('predictions:',res)
print('pred:',np.argmax(res,axis=1))




