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
        height, width, channels = frame.shape
        blank_image = numpy.zeros((height, width, channels), numpy.uint8)
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
        self.height2, self.width2, _ = self.img.shape

    def draw(self, frame):
        if self.do_stop:
            return frame

        logging.debug("image filter")

        frame = FilterColor.add_color(frame, (255, 178, 0))

        # add an alpha channel
        b_channel, g_channel, r_channel = cv2.split(frame)
        alpha_channel = numpy.ones(b_channel.shape, dtype=b_channel.dtype) * 50
        frame = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        height, width, channels = frame.shape

        # resize to the frame size and center the image
        blank = numpy.zeros((height, width, channels), numpy.uint8)
        x, y = (width - self.width2) // 2, (height - self.height2) // 2
        blank[y:y+self.height2, x:x+self.width2] += self.img

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
        frame = cv2.flip(frame, 1)
        frame = cv2.putText(frame, self.text, (x, y), font, fontScale, color, thickness, cv2.LINE_AA)
        frame = cv2.flip(frame, 1)
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
    """
    https://medium.com/@stepanfilonov/tracking-your-eyes-with-python-3952e66194a6
    """

    face_cascade = cv2.CascadeClassifier('img/haarcascade_frontalface_default.xml')

    def __init__(self, frame):
        super(FilterBlur, self).__init__(duration=0)
        self.do_stop = False
        self.last_frame = frame
        self.debug = False

    @staticmethod
    def pixelize(frame, faces):
        """Pixelize each face."""

        height, width, channels = frame.shape

        # https://stackoverflow.com/a/55509210
        pixelized = frame.copy()
        pixelized = cv2.resize(pixelized, (width // 12, height // 12), interpolation=cv2.INTER_LINEAR)
        pixelized = cv2.resize(pixelized, (width, height), interpolation=cv2.INTER_NEAREST)

        blank = numpy.zeros((height, width, channels), numpy.uint8)

        for x, y , w, h in faces:
            blank[y:y+h,x:x+w] += pixelized[y:y+h,x:x+w]

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
    def areas_overlap(a, b):
        (x1, y1, w1, h1) = a
        (x2, y2, w2, h2) = b
        dx = min(x1+w1, x2+w2) - max(x1, x2)
        dy = min(y1+h1, y2+h2) - max(y1, y2)
        return dx >= 0 and dy >= 0

    @staticmethod
    def merge_faces(faces, width, frame):
        """
        If two faces overlap, merge them.

        This is way too complicated, but overlapping areas gets blurred twice
        otherwise. A much simpler workaround would be to keep the largest face.
        """

        if len(faces) == 1:
            return faces

        while True:
            overlap = False
            merged_faces = faces
            for i, a in enumerate(faces):
                (x1, y1, w1, h1) = a
                for j, b in enumerate(faces):
                    if j >= i:
                        continue
                    (x2, y2, w2, h2) = b
                    overlap = FilterBlur.areas_overlap(a, b)
                    if overlap:
                        # remove both faces
                        for x in [a, b]:
                            try:
                                merged_faces.remove(x)
                            except ValueError:
                                pass
                        # add merged faces
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = max(x1+w1, x2+w2) - x
                        h = max(y1+h1, y2+h2) - y
                        merged_faces.append((x, y, w, h))
                        break
                if overlap:
                    break

            faces = merged_faces
            if not overlap:
                break

        return faces

    @staticmethod
    def detect_faces(frame, cascade):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cascade.detectMultiScale(gray_frame, 1.3, 5)

    @staticmethod
    def increase_face_area(faces, margin=80):
        return [ (x - margin // 2, y - margin // 2, w + margin, h + margin) for x, y, w, h in faces ]

    @staticmethod
    def mark_faces(frame, faces, rgb=(255, 255, 0)):
        """Add a rectangle around each face."""

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), rgb, 2)

    def draw(self, frame):
        faces = FilterBlur.detect_faces(frame, FilterBlur.face_cascade)
        # setting this condition to True prevent an image with no face blurred
        # from being returned
        if len(faces) == 0 and True:
            frame = self.last_frame
        else:
            _, width, _ = frame.shape
            faces = FilterBlur.increase_face_area(faces)
            faces = FilterBlur.merge_faces(faces, width, frame)
            frame = FilterBlur.pixelize(frame, faces)
            if self.debug:
                FilterBlur.mark_faces(frame, faces)
            self.last_frame = frame

        return frame
