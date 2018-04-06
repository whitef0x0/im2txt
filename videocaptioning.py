from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from picamera.array import PiRGBArray
from picamera import PiCamera

from PIL import Image
from io import BytesIO
import pyttsx3

import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

tf.logging.set_verbosity(tf.logging.ERROR)

VOCAB_FILE="/home/pi/project2/model_data/word_counts.txt"
MODEL_FILE="/home/pi/project2/model_data/model.ckpt-2000000"

def encode(npdata):
  img = Image.fromarray(npdata)
  output = BytesIO()
  img.save(output, "jpeg")
  image = output.getvalue()
  output.close()
  return image

def run():
  engine = pyttsx3.init()

  print("Building inference graph")
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               MODEL_FILE)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(VOCAB_FILE)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)


    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)


    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 10

    print("Initialized camera")
    rawCapture = PiRGBArray(camera, size=(640, 480))
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
      image = encode(frame.array)
      captions = generator.beam_search(sess, image)
      
      if captions:
        caption = captions[0]
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        engine.say(sentence)
        engine.runAndWait()
  

      rawCapture.truncate(0)


if __name__ == "__main__":
  run()
