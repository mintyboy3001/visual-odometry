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

        start_time = time.time()
        frame_count = 0
        prev_count = 0

        while True:
            _, frame = cap.read()

            buffer.put(frame)
            # Calculate FPS
            frame_count += 1
            cv.putText(frame, f'Frame: {frame_count:>3}', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            fcopy = copy.deepcopy(frame)
            # Display FPS on the frame
            if time.time() - start_time >= 1.0:
               
                prev_count = frame_count
                start_time = time.time()
                frame_count = 0
            
            
            cv.putText(fcopy, f'FPS: {prev_count:>3}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Live Feed', fcopy)

            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv.destroyAllWindows()
                print("Child 1 exiting")
                exit()

    
    def handle_buffer(self,buffer,skips = 2):

        start_time = time.time()
        frame_count = 0
        #fps = 30 / (skips+1)
        while True:
            try:
                frame = buffer.get()
                # print(type(batch))
                # print(type(batch[0]))
                # print(batch[0].shape)
                # Simulate expensive computation
                # frame_count += 1
                # fcopy = copy.deepcopy(frame)

                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                cv.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow('lagged',frame)

                for _ in range(skips):
                    buffer.get()                
                
            except:
                pass

            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                print("Child 2 exiting")
                exit()
        


    
    def spin(self):
        max_buffer_size = 300
        buffer = multiprocessing.Queue(maxsize=max_buffer_size)
        # Create separate processes for reading and processing
        camera_process = multiprocessing.Process(target=self.read_camera,args=(buffer,))
        buffer_process = multiprocessing.Process(target=self.handle_buffer,args=(buffer,))

        # Start both processes
        camera_process.start()
        buffer_process.start()

        camera_process.join()
        buffer.put(None)
        buffer_process.terminate()
        print("Trying to destroy windows")
        cv.destroyAllWindows()


if __name__ == "__main__":
    cam = CameraBuffer()
    cam.spin()



