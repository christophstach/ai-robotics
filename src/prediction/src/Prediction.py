#!/usr/bin/env python

import numpy as np
import rospy
import tensorflow as tf
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from tensorflow.keras.models import load_model


class Prediction:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.publisher_specific_number = None

        # filepath = dirname(realpath(__file__)) + '/model.h5'
        filepath = '/home/christoph/PycharmProjects/robotik-tomato-long-antbear/src/excercise_1/src/model.h5'
        self.model = load_model(filepath)
        self.graph = tf.get_default_graph()

    def subscribe_specific_image(self):
        rospy.loginfo('subscribing to specific image')
        rospy.Subscriber('/camera/output/specific/compressed_img_msgs', CompressedImage, self.callback_specific_image)

    def subscribe_specific_check(self):
        rospy.loginfo('subscribing to specific check')
        rospy.Subscriber('/camera/output/specific/check', Bool, self.callback_specific_check)

    def publish_specific_number(self):
        rospy.loginfo('publishing to specific number')
        self.publisher_specific_number = rospy.Publisher('/camera/input/specific/number', Int32, queue_size=1)

    def subscribe_random_image(self):
        rospy.loginfo('subscribing to random image')
        rospy.Subscriber('/camera/output/random/compressed_img_msgs', CompressedImage, self.callback_random_image)

    def subscribe_random_number(self):
        rospy.loginfo('subscribing to random number')
        rospy.Subscriber('/camera/output/random/number', Int32, self.callback_random_number)

    def callback_specific_image(self, msg):
        with self.graph.as_default():
            img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            batch = np.expand_dims(img, axis=0)
            batch = batch.reshape(-1, 28, 28, 1)
            batch = batch / 255.0

            pred_one_hot = self.model.predict(batch)
            pred = np.argmax(pred_one_hot)

            rospy.loginfo('Prediction is: %s' % pred)

            if self.publisher_specific_number:
                self.publisher_specific_number.publish(pred)

    def callback_specific_check(self, msg):
        rospy.loginfo('Correct: %s\n' % msg.data)

    def callback_random_image(self, msg):
        with self.graph.as_default():
            img = self.cv_bridge.compressed_imgmsg_to_cv2(msg)
            batch = np.expand_dims(img, axis=0)
            batch = batch.reshape(-1, 28, 28, 1)
            batch = batch / 255.0

            pred_one_hot = self.model.predict(batch)
            pred = np.argmax(pred_one_hot)

            rospy.loginfo('Prediction is: %s\n' % pred)

    def callback_random_number(self, msg):
        rospy.loginfo('Number is: %s' % msg.data)


def main():
    try:
        # register node
        rospy.init_node('prediction', anonymous=False)
        rospy.loginfo('Starting prediction...')

        # init CameraPseudo
        prediction = Prediction()

        prediction.subscribe_specific_image()
        prediction.subscribe_specific_check()
        prediction.publish_specific_number()

        prediction.subscribe_random_image()
        prediction.subscribe_random_number()

        print('\n')

        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
