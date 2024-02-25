import socket

def send_signal(host='127.0.0.1', port=12345):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(b'Hello, world')

if __name__ == '__main__':
    send_signal()