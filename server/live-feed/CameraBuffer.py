import multiprocessing
import numpy as np
import cv2 as cv
import time
import copy



class CameraBuffer:
    
    def __init__(self,stream_url='tcp://192.168.0.31:8123'):
        self.stream_url = stream_url
        self.frame_shape = (1280, 720, 3)  
      
     
        

    def read_camera(self,buffer):
        batch = []
        cap = cv.VideoCapture(self.stream_url,cv.CAP_FFMPEG)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            _, frame = cap.read()
            batch_size = 10
            
            
            cv.imshow('frame', frame)
            batch.append(frame)

            if len(batch) == batch_size:
                buffer.put(batch)
                batch = []

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    
    def handle_buffer(self,buffer):
        while True:
            try:
                batch = buffer.get()
                # print(type(batch))
                # print(type(batch[0]))
                # print(batch[0].shape)
                # Simulate expensive computation
                #time.sleep(0.25)
                for frame in batch:
                    cv.imshow('lagged',frame)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                pass
        


    
    def spin(self):
        max_buffer_size = 120
        buffer = multiprocessing.Queue(maxsize=max_buffer_size)
        # Create separate processes for reading and processing
        camera_process = multiprocessing.Process(target=self.read_camera,args=(buffer,))
        buffer_process = multiprocessing.Process(target=self.handle_buffer,args=(buffer,))

        # Start both processes
        camera_process.start()
        buffer_process.start()

        camera_process.join()
        buffer.put(None)
        buffer_process.join()
        cv.destroyAllWindows()


if __name__ == "__main__":
    cam = CameraBuffer()
    cam.spin()



