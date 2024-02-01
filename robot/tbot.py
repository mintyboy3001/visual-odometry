from trilobot import Trilobot
import socket
import subprocess

server_address = ('0.0.0.0', 12345)

class BotClient():

    def __init__(self):
        #self.tbot = Trilobot()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(server_address)

        command = "rpicam-vid -t 0 --exposure sport --width 1280 --height 720 --nopreview --inline --listen -o tcp://0.0.0.0:8123".split(" ")
        #start rpicam stream
        self.process = subprocess.Popen(command)

    def spin(self):
        self.socket.listen(1)
        print(f"Waiting for incoming connection on {server_address}...")

        while True and self.process.poll() is None:
            # Wait for a connection
            client_socket, client_address = self.socket.accept()
            
            try:
                print(f"Connection from {client_address}")
                
                # Receive and print data from the client
                data = client_socket.recv(1024)
                if data:
                    print(f"Received data: {data.decode()}")
            finally:
                # Clean up the connection
                client_socket.close()
        
        print("Closed or camera stream died")



if __name__ == "__main__":
    client = BotClient()
    client.spin()
