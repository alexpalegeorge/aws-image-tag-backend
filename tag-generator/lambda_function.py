import json
import boto3
import urllib.parse
import cv2
import numpy as np
import os

yolo_path  = "/tmp/"
confthres = 0.3
nmsthres = 0.1

def lambda_handler(event, context):
    
    yolo_bucket_name = "fit5225-tagtag-yolo-bucket"

    labelsPath= "coco.names"
    local_labelsPath = '/tmp/' + labelsPath

    cfgpath= "yolov3-tiny.cfg"
    local_cfgpath = '/tmp/' + cfgpath

    wpath= "yolov3-tiny.weights"
    local_wpath = '/tmp/' + wpath

    # copy yolo items to working directory
    s3_yolo = boto3.resource('s3')
    s3_yolo.Bucket(yolo_bucket_name).download_file(labelsPath, local_labelsPath)
    s3_yolo.Bucket(yolo_bucket_name).download_file(cfgpath, local_cfgpath)
    s3_yolo.Bucket(yolo_bucket_name).download_file(wpath, local_wpath)

    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    
    
    s3 = boto3.client('s3')
    
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        fileObj = s3.get_object(Bucket=bucket, Key=key)
        file_content = fileObj["Body"].read()
        
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
        
    np_array = np.fromstring(file_content, np.uint8)
    cv_image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
    image = cv_image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load the neural net.  Should be local to this method as its multi-threaded endpoint
    nets = load_model(CFG, Weights)

        # return the list
    tags = do_prediction(image, nets, Lables)
    url = "https://" + bucket + ".s3.amazonaws.com/" + key
    
    add_to_db(key, tags)

    return {
        'statusCode': 200,
        'url': url,
        'data': json.dumps(tags)
        }
        

def add_to_db(key, tags):
    
    dynamodb = boto3.resource('dynamodb')
    dynamo_table = dynamodb.Table('tagtag-sorted')
    for t in tags:
        dynamo_table.put_item(
            Item = {
                'tag': t,
                'image': key
            }
        )
    

def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath=os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS

def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image,net,LABELS):

    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]
            # print("confidence: ", confidence)

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # TODO Prepare the output as required to the assignment specification
    # ensure at least one detection exists
    tag_list = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            tag_list.append(LABELS[classIDs[i]])
    else:
        tag_list.append('no objects')
    return tag_list
