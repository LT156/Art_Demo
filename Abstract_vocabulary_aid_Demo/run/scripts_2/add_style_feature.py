import pandas as pd
save_path = r'F:\work\Image_emotion_analysis\artemis-master\RAIVS_out\step0\image_emotion_histogram_style.csv'
style_file = r'F:\work\Image_emotion_analysis\artemis-master\RAIVS\scripts_2\additional_files\style_class.txt'
file_path = r'F:\work\Image_emotion_analysis\artemis-master\artemis\data\image-emotion-histogram.csv'
df = pd.read_csv(file_path)

style_dict = dict()
try:
  with open(style_file,'r',encoding = 'utf-8') as f:
    temp = f.read()
    line_list = temp.splitlines()
    lines = len(line_list)
    for line in line_list:
        items = line.split(' ')
        style_dict[items[1]] = items[0]
except Exception as e:
  print('遇到错误：\n',e)


style_ids = df['art_style'].apply(lambda x:style_dict[x])
df.insert(1,'style_id',style_ids)
df.to_csv(save_path)