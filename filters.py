from abc import ABC, abstractmethod
import argparse
import datetime
import cv2
import logging
import numpy

class Filter:
    def __init__(self, duration=1.0):
        self.start_time = datetime.datetime.now()
        self.duration = duration
        self.do_stop = False
        logging.debug("applying filter")

    def stop(self):
        self.do_stop = True

    def done(self):
        diff = datetime.datetime.now() - self.start_time
        done = (self.duration > 0 and diff.total_seconds() >= self.duration) or self.do_stop
        if done:
            logging.debug("filter applied")
        return done

    @abstractmethod
    def draw(self, frame):
        pass

class FilterColor(Filter):
    def __init__(self, color=(255, 0, 0)):
        super(FilterColor, self).__init__()
        self.color = color

    @staticmethod
    def add_color(frame, color):
        width, height, channels = frame.shape
        blank_image = numpy.zeros((width, height, channels), numpy.uint8)
        blank_image[:,:] = color
        frame = cv2.addWeighted(frame, 0.7, blank_image, 0.3, 0)
        return frame

    def draw(self, frame):
        logging.debug("color filter")
        frame = FilterColor.add_color(frame, self.color)
        return frame

class FilterAddImage(Filter):
    def __init__(self, path):
        super(FilterAddImage, self).__init__(duration=2)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logging.error(f"image file '{path}' doesn't exist")
            self.do_stop = True
            return

        self.img = cv2.resize(img, (150, 150))
        self.width2, self.height2, _ = self.img.shape

    def draw(self, frame):
        if self.do_stop:
            return frame

        logging.debug("image filter")

        frame = FilterColor.add_color(frame, (255, 178, 0))

        # add an alpha channel
        b_channel, g_channel, r_channel = cv2.split(frame)
        alpha_channel = numpy.ones(b_channel.shape, dtype=b_channel.dtype) * 50
        frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        width, height, channels = frame.shape

        # resize to the frame size and center the image
        blank = numpy.zeros((width, height, channels), numpy.uint8)
        x, y = (width - self.width2) // 2, (height - self.height2) // 2
        blank[x:x+self.width2, y:y+self.height2] += self.img

        # create a mask from the given image
        img2gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # apply this mask to the frame: this region is now black
        frame_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)

        # add the image
        frame = cv2.add(frame_bg, blank)

        return frame

class FilterAddText(Filter):
    def __init__(self, text):
        super(FilterAddText, self).__init__(duration=0)
        self.text = text
        self.do_stop = False

    def draw(self, frame):
        frame = FilterColor.add_color(frame, (255, 178, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 255, 255)
        thickness = 3
        textsize = cv2.getTextSize(self.text, font, 1, 2)[0]
        x = (frame.shape[1] - textsize[0]) // 2
        y = (frame.shape[0] + textsize[1]) // 2 - 10
        frame = cv2.putText(frame, self.text, (x, y), font, fontScale, color, thickness, cv2.LINE_AA)
        return frame

class FilterVideo(Filter):
    def __init__(self, path):
        super(FilterVideo, self).__init__(duration=0)
        self.video = cv2.VideoCapture(path)
        if not self.video.isOpened():
            logging.error(f"video file '{path}' doesn't exist")
            self.do_stop = True

    def draw(self, frame):
        if self.do_stop:
            return frame

        height, width, _ = frame.shape
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (width, height))
        return frame

class FilterBlur(Filter):
    face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')

    def __init__(self, frame):
        super(FilterBlur, self).__init__(duration=0)
        self.do_stop = False
        self.last_frame = frame

    @staticmethod
    def blur(frame, faces):
        height, width, channels = frame.shape

        # https://stackoverflow.com/a/55509210
        pixelized = frame.copy()
        pixelized = cv2.resize(pixelized, (width // 12, width // 12), interpolation=cv2.INTER_LINEAR)
        pixelized = cv2.resize(pixelized, (width, height), interpolation=cv2.INTER_NEAREST)

        blank = numpy.zeros((height, width, channels), numpy.uint8)

        margin = 80
        for x, y , w, h in faces:
            x -= margin // 2
            y -= margin // 2
            w += margin
            h += margin
            blank[y:y+h,x:x+w] += pixelized[y:y+h,x:x+w]
            if False:
                cv2.rectangle(blank, (x,y),(x+w,y+h), (255, 255, 0), 2)

        # create a mask from the given image
        img2gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # apply this mask to the frame: this region is now black
        frame_bg = cv2.bitwise_and(frame, frame, mask = mask_inv)

        # add the image
        frame = cv2.add(frame_bg, blank)
        return frame

    @staticmethod
    def detect_faces(img, cascade):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_frame, 1.3, 5)
        if len(faces) == 0:
            # setting this variable to True prevent an image with no face
            # blurred from being returned
            if False:
                return None
            else:
                return img
        frame = FilterBlur.blur(img, faces)
        return frame

    def draw(self, frame):
        face_frame = FilterBlur.detect_faces(frame, FilterBlur.face_cascade)
        if face_frame is None:
            frame = self.last_frame
        else:
            frame = face_frame
            self.last_frame = frame
        return frame
