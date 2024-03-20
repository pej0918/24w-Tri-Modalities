## 현재 수정 중 

```
python train.py --we_path your/we/path --data_path your/data/path --token_projection projection_net
```

여기에서 token_projection을 

1. gated를 사용하면 (everything at once 원본)
토큰 차원이 이런식으로 맞춰짐  
text_raw_embed = self.extract_text_tokens(text) # [16, 30, 4096]  
video_raw_embed = self.extract_video_tokens(video) # [16, 4096]  
audio_raw_embed = self.extract_audio_tokens(audio, nframes) # [16, 80, 4096]  
이걸 고려해서 임베딩 시켜야할 것 같은데...avlnet 어떻게 했는지 찾아보자  

2. projection_net을 사용하면 (한결이가 올린 거)
video = self.video_projection(video)  
RuntimeError: mat1 and mat2 shapes cannot be multiplied (16x4096 and 2048x512)  
여기에서 이 에러 뜸 ... 내일 해결하자^^


어쩄든 이 차원맞추면 FusionTransformer에서 forward의 token 차원 맞춰지는 거고  
이 차원에 맞춰서 cls_token 만들고 앞에 concat하고  
Q, K, V 세개 섞어서 block(fusionblock) 돌리면 될 것 같당  
