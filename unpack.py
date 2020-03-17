import cv2

def main():
    video_path = '/mnt/fileserver/shared/datasets/cameras/Odessa/Duke_on_the_left/fragments/child/child_set005_00:16:20-00:20:00.mp4'
    cap = cv2.VideoCapture(video_path)
    i = 1
    str_len = 4
    while True:
        res, img0 = cap.read()  # BGR
        if not res:
            return
        name = str(i)
        name = '0' * (str_len - len(str(name))) + name + '.png'
        cv2.imwrite('./dataset/child1620/images/' + name, img0)

if __name__ == '__main__':
    main()
