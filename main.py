from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from threading import Lock, Event
import base64
import cv2
import numpy as np
from picamera2 import PiCamera2

thread_event = Event()
clients = set()
image = None

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=async_mode, logger=True, engineio_logger=True)
thread = None
thread_lock = Lock()
camera = None

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

@socketio.event
def connect():
    global clients, image
    clients.add(request.sid)
    start_background_thread()
    if image:
        retval, buffer = cv2.imencode('.png', image)
        encoded = base64.b64encode(buffer)
        emit('image', {'ext': "png", 'data': encoded})

@socketio.on('disconnect')
def on_disconnect(reason):
    global clients
    clients.discard(request.sid)
    if len(clients) == 0:
        stop_background_thread()
    print('Client disconnected', request.sid, reason)


def start_background_thread():
    global thread, camera
    with thread_lock:
        if thread is None:
            thread_event.set()
            camera = PiCamera2()
            config = camera.create_still_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            camera.configure(config)
            camera.start()
            socketio.sleep(1)
            thread = socketio.start_background_task(background_thread, thread_event)

def stop_background_thread():
    global thread, camera
    thread_event.clear()
    with thread_lock:
        if thread is not None:
            thread.join()
            thread = None
            camera.close()


def background_thread(event):
    global thread, image
    try:
        while event.is_set():
            socketio.sleep(.2)
            new_image = capture_image_cv2()
            if image is None:
                retval, buffer = cv2.imencode('.png', new_image)
                encoded = base64.b64encode(buffer)
                socketio.emit('image', {'ext': "png", 'data': encoded})
            elif diff_image := circle_image_differences(image, new_image):
                retval, buffer = cv2.imencode('.png', diff_image)
                encoded = base64.b64encode(buffer)
                socketio.emit('image', {'ext': "png", 'data': encoded})
            image = new_image
    finally: 
        event.clear()
        thread = None

def capture_image_cv2():
    global camera

    image_rgb = camera.capture_array()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr

def circle_image_differences(img1, img2, min_area=50):
    """
    Takes two images and returns an image with circles
    drawn around all detected differences.
    
    :param img1: first image
    :param img2: second image
    :param min_area: minimum contour area to count as a difference
    :return: image with circles around differences (numpy array)
    """

    if img1 is None or img2 is None:
        raise ValueError("One or both images could not be loaded")

    # Resize second image if sizes differ
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute absolute difference
    diff = cv2.absdiff(gray1, gray2)

    # Threshold the difference image
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Dilate to merge nearby differences
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw circles around differences
    should_output = False

    output = img2.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(output, center, radius + 5, (0, 0, 255), 2)
            should_output = True

    if should_output:
        return output
    else:
        return False

if __name__ == '__main__':
    socketio.run(app)
