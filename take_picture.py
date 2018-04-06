#!/usr/bin/env python

import serial
import picamera
import sys, os, subprocess
import requests, json
import time, datetime

import tensorflow as tf
import inference_wrapper
import configuration

from inference_utils import caption_generator
from inference_utils import vocabulary

SERVER_URL = 'http://ec2-18-191-1-128.us-east-2.compute.amazonaws.com/images'
UPLOAD_IMAGE_URI = 'images'
PHOTOFORMAT = 'jpeg'

VOCAB_FILE = '/home/pi/project2/model_data/word_counts.txt'
MODEL_FILE = '/home/pi/project2/model_data/model.ckpt-2000000'


def takePicture(filename):
  with picamera.PiCamera() as camera:
    camera.resolution = (640, 480) #(1920, 1080)
    camera.capture(filename + '.' + PHOTOFORMAT, format=PHOTOFORMAT)
    camera.close()
    print("Photo captured and saved ...")
    return filename + '.' + PHOTOFORMAT

def timestamp():
  tstring = datetime.datetime.now()
  print("Filename generated ...")
  return tstring.strftime("%Y%m%d_%H%M%S")

def deleteFile(filename):
  os.system("rm " + filename)
  print("File: " + filename + " deleted ...")

def uploadPicture(filename, caption, coords):
  filePath = './' + filename + '.' + PHOTOFORMAT

  print("Uploading " + filename + " to AWS Server")
  
  headers = {'caption': caption, 'coordinates': coords} 
  files = {'file': open(filePath, 'rb')}
  url = SERVER_URL

  #try: 
  r = requests.post(url, files=files, headers=headers)
  print(r.status_code)
  print(r.text)
  #r.raise_for_status()
  #except Exception as e:
  #  print("ERROR: File upload failed")
  #  print(e.args[0])
  #else:
  #  print("File upload succeeded!")

def getLocation():
  send_url = 'http://freegeoip.net/json'
  r = requests.get(send_url)
  j = json.loads(r.text)
  lat = j['latitude']
  lon = j['longitude']
  return str(lat) + ',' + str(lon)

def generate_caption_local(filename, sess, generator, vocab):
  with tf.gfile.GFile(filename, "r") as f:
    image = f.read()

  captions = generator.beam_search(sess, image)
  
  if captions:
    caption = captions[0]
    sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
    sentence = " ".join(sentence)
    return sentence
  return ""
  
def main():
  g = tf.Graph()  
  print("created tf graph")
  
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(), MODEL_FILE)
  print("loaded model into tf graph")  

  g.finalize()

  #Create vocab
  vocab = vocabulary.Vocabulary(VOCAB_FILE)
  print("loaded vocab model")
  with tf.Session(graph=g) as sess:
    print("started tf session")
    restore_fn(sess)

    generator = caption_generator.CaptionGenerator(model, vocab)
    print("created caption generator")
    serialData = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    
    print("start serial conn")  
    while True:
      import time
      input = serialData.read()
      print("waiting for take photo message")
      while(input != b'P'):
        input = serialData.read()
      serialData.write(b'p')
      print("photo message has been received")


      print("waiting for GPS message")
      input = serialData.read()
      while(input != b'G'):
        input = serialData.read()
    
      serialData.write(b'g')
      print("GPS message has been received")    

      while(serialData.in_waiting == 0):
        continue; 
      latitude = ""
      input = serialData.read()
      
      while(input != b'\n'):
        print(input)
        if input != b'\x00':
          latitude = latitude + input.decode("utf-8")
        input = serialData.read()  

      print("latitude: "+latitude) 
      while(serialData.in_waiting == 0):
        continue;
      input = serialData.read()
      
      longitude = "" 
      while(input != b'\n'):
        if input != b'\x00':
          longitude = longitude + input.decode("utf-8")
        input = serialData.read()

      print("longitude: "+longitude)
      #Generate filename from current time
      filename = timestamp()
      
      #Capture photo
      file = takePicture(filename)
      
      filePath = "./" + filename + "." + PHOTOFORMAT  
      caption = generate_caption_local(filePath, sess, generator, vocab)
      coords = latitude + ',' + longitude 
      
      subprocess.call("../speech.sh "+caption, shell=True)
      print("Generated caption for photo is: " + caption)
      #Upload photo
      uploadPicture(filename, caption, coords)

      #Delete local file
      deleteFile(filename + '.' + PHOTOFORMAT)

    print("Done\n\n")

if __name__ == '__main__':
  main() 
