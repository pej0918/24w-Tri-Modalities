import pandas as pd
import os
from pytube import YouTube
'''
howto100M의 모든 데이터를 다 다운로드 할 수 없어서 일단 일부의 데이터만 직접 크롤링하여, 전처리 코드 및 loader 코드를 짜고 있습니다.
일단은, 이렇게 전처리 코드를 짜지만 이후, 는 일반화된 전처리 코드로 변경해서 올리고, 저희는 아마 s3d를 이용해 이미 추출한 feature를 사용해야 하지 않을까 고민중입니다.
일부의 데이터셋을 크롤링해서 그것만 맞춘 csv를 만들어서 처리하도록 했습니다 (테스트 용)
'''

# pytube를 이용한 데이터셋 크롤링 코드
def crawling(DownloadFolder ,csv_path):
    df = pd.read_csv(csv_path)
    for video_id in df['video_id']:
      link='https://www.youtube.com/watch?v={}'.format(video_id)
      try:
          video = YouTube(link)
          clip = video.streams.get_highest_resolution()
          output_path = os.path.join(DownloadFolder, video_id)
          if not os.path.exists(output_path):
              os.makedirs(output_path)
          clip.download(output_path)
          print(f"Downloaded: {video_id}")

      except Exception as e:
          print(f"Error downloading video {video_id}: {str(e)}")


# 실제로 받은 데이터셋 매칭 -> path정보 추가해서 이후 전처리에 사용
def matching_dataset(path, video_path):
    df = pd.read_csv(path)
    video_names = os.listdir(video_path)
    df_f = df[df['video_id'].isin(video_names)]
    df_f.to_csv(f"TK/data/HowTo100M_v1_{len(video_names)}.csv")


# 크롤링
if __name__ == "__main__":
    csv_path = "TK/data/HowTo100M_v1.csv"
    DownloadFolder = 'Tk/Howto100M/howto100m_video/videos'
    crawling(DownloadFolder,csv_path)

    # matching 용
    # path = "TK/data/HowTo100M_v1.csv"
    # video_path = "TK/Howto100M/howto100m_video/videos"
    # matching_dataset(path, video_path)