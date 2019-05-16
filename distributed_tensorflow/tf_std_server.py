"""Run a standard tensorflow server."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# if not with kubernetes
import os
import json
os.environ["TF_CONFIG"] = json.dumps({
    "cluster": {
        "worker": ["hostname1:5000", "hostname2:5000"]
    },
   "task": {"type": "worker", "index": 0}
})

def main(unused_argv):
  # Contrib ops are lazily loaded. So we touch one contrib module to load them
  # immediately.
  to_import_contrib_ops = tf.contrib.resampler

  # Load you custom ops here before starting the standard TensorFlow server.

  # Start and join the standard TensorFlow server.
  tf.contrib.distribute.run_standard_tensorflow_server().join()


if __name__ == "__main__":
  tf.app.run()