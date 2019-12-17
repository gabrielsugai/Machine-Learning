# import required packages
import cv2
import time
import numpy as np
import os

#Definições globais
#camera = cv2.VideoCapture("../Video_Teste_OpenCV.mp4")

# read class names from text file
classes = None
with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

#Funcoes para manipular os dados
# function to get the output layer names
# in the architecture
def get_output_layers(net):

    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, "%.4f" % (confidence), (x-10,y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


l_imagens = []
#l_imagens = os.listdir("C:/Users/CPU/Desktop/Python_OpenCV_Robotica/Yolo/imagens/")
l_imagens = os.listdir()
print(l_imagens)
#for i in range(0, (len(l_imagens))-x):
#    if l_imagens[i] == 'rede_yolo.py':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'rede_yolo2.py':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'teste_cvlib.py':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'yolov3.cfg':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'yolov3.txt':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'yolov3.weights':
#        del(l_imagens[i])
#        x+=1
#    elif l_imagens[i] == 'yolov3_base.txt':
#        del(l_imagens[i])
#        x+=1


while True:
    print(l_imagens)
    for xxx in range(0, (len(l_imagens))):
#        print(l_imagens[xxx])
        image = cv2.imread(l_imagens[xxx])
        original = image
        img_final = cv2.blur(original, (20,20))
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
        #blob = cv2.resize(image, (416,416), cv2.INTER_LINEAR)
        # set input blob for the network
        net.setInput(blob)
        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    # apply non-max suppression
                    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
                    # go through the detections remaining
                    # after nms and draw bounding box
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            # print("Achei: ",classes[class_ids[i]], " Chance: %.4f" % confidences[i])
            if classes[class_ids[i]] == "person":
                ix,jy = int(x), int(y)
                xw = int((x+w))
                yh = int((y+h))
                for i in range(jy, yh):
                    for j in range(ix, xw):
                        img_final[i,j] = original[i,j]
                        #        print(box)
                        # else:
                        #  continue
                        #draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                        #cv2.imshow(l_imagens[xxx], img_final)
        salvar_novaimagem = l_imagens[xxx].split('.')
        cv2.imwrite(salvar_novaimagem[0]+"nova."+salvar_novaimagem[1], img_final) 
        # display output image
        # img = cv2.resize(image, (800,800))
        # cv2.imshow("object detection", img)
        # cv2.imwrite("object detection.jpg", image)
        #cv2.imshow("object detection", image)
        #time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # time.sleep(0.5)

 # save output image to disk
# cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()
