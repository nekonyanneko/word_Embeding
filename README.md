# word2vec

## **word2vecの環境を構築する。**
**1. Dockerで構築するならDockerfileは用意してあるため、そちらを読み込む**
以下は、よく使うコマンド
```
docker build -t word2vec/word2vec:1.0 .
docker ps -a
docker rm psID
docker images
docker rmi imID
docker run --name word2vec -it psID /bin/bash
```
```
docker run -v [ホストディレクトリの絶対パス]:[コンテナの絶対パス] [イメージ名] [コマンド]
docker run -it -v /your/path:/tmp/ IMAGE_ID /bin/bash
```
