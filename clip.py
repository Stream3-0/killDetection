import cv2
import os 
import datetime
import ffmpy 

template = cv2.imread('./skull.png')

class Clip:
    def __init__(self, file):
        self.rawPath = f"./clips/raw/{file}"
        self.file = file
        self.fileName = self.remove_extension(file)
        
        if not (os.path.exists(self.rawPath)):
            raise FileNotFoundError(file)

    
    @classmethod
    def from_url(cls, fileName, url):
        rawPath = f"./clips/raw/{fileName}"

        ff = ffmpy.FFmpeg(
            inputs = {url: None},
            outputs= {rawPath: "-c copy"}
        )

        print(ff.cmd)

        ff.run()
        
        return cls(fileName)


    @staticmethod
    def remove_extension(file):
        return file[:file.index(".")]


    @staticmethod
    def get_frame_from_file(file):
        return int(file[file.index("_")+1:file.index(".")])


    def split_video_into_frames(self):
        self.outputPath = f"./clips/frames/{self.fileName}"
        
        if os.path.exists(self.outputPath):
            return
        else:
            os.makedirs(self.outputPath)

        capture = cv2.VideoCapture(self.rawPath)

        frameNr = 0
        skip = 10
        
        while (True):
            success, frame = capture.read()

            if success:
                if frameNr % skip == 0:
                    print("Saving frame:", frameNr)
                    cv2.imwrite(f'{self.outputPath}/frame_{frameNr}.jpg', frame)
            else:
                break
        
            frameNr += 1

        capture.release()


    @staticmethod
    def is_template_in_image(img, templ):

        result = cv2.matchTemplate(img, templ, cv2.TM_SQDIFF)

        min_val = cv2.minMaxLoc(result)[0]

        thr = 40000000

        return min_val <= thr


    def identify_clips(self):
        self.split_video_into_frames()
        
        video = cv2.VideoCapture(self.rawPath);
        
        self.fps = round(video.get(cv2.CAP_PROP_FPS))
        print("FPS", self.fps)
                
        kills = []
        
        for file in os.listdir(self.outputPath):
            print("Processing frame:", file)
            image = f"{self.outputPath}/{file}"

            if self.is_template_in_image(cv2.imread(image), template):
                frame = self.get_frame_from_file(file)
                print("Found kill at frame", frame)
                timestamp = round(frame / self.fps)
                kills.append(timestamp - 3) 
          
        return self.clean_stamps(kills) 


    @staticmethod
    def clean_stamps(stamps):
        stamps = sorted(stamps)
        i = 0
        while (i < len(stamps)):
            j = i + 1
            while (j < len(stamps)):
                if stamps[j] <= stamps[i] + 5:
                    stamps.pop(j)
                    j -= 1
                j+=1
            i+=1

        return [str(datetime.timedelta(seconds=seconds)) for seconds in stamps]


if __name__ == "__main__":
    clip = Clip.from_url("clip4.mp4", "https://media.thetavideoapi.com/video_vbcun7zncmi2zgcabg4zx8rgyt/master.m3u8")
    #clip = Clip("clip3.mp4")
    print(clip.identify_clips())