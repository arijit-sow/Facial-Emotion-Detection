import os
def main():
    i=1
    path = "E:/git file/Emotion_Detection_CNN/extra_data/happy_download_image/"
    for filename in os.listdir(path):
        my_dest =  str(i) + ".jpg"
        my_source = path+filename
        my_dest = path + my_dest
        os.rename(my_source , my_dest)
        i+=1
if __name__ == '__main__':
    main()