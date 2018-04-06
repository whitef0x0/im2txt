from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

tf.logging.set_verbosity(tf.logging.ERROR)

VOCAB_FILE="/home/pi/project2/model_data/word_counts.txt"
MODEL_FILE="/home/pi/project2/model_data/model.ckpt-2000000"

def generateCaption(filepath, tf_sess):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               MODEL_FILE)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(VOCAB_FILE)

  filenames = []
  filenames.extend(tf.gfile.Glob(filepath))

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    for filename in filenames:
      print("\n\n\nfilename: %s\n", filename)
      with tf.gfile.GFile(filename, "r") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      #for i, caption in enumerate(captions):
      # Ignore begin and end words.
      #sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
      #sentence = " ".join(sentence)
      #print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

      if captions:
        caption = captions[0]
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        return sentence
      else:
        return ""

#if __name__ == "__main__":
#  generateCaption(IMAGE_FILE)
