

# build docker

```bash
# run with sudo if needed
docker build docker/23-01/build.sh
```


# Run OWL

1. run the docker container
run `docker/23-01/run.sh`

2. Install OWL dependencies in the container
```bash
cd /nanoowl
python3 setup.py develop --user

# build tensorrt engine
mkdir -p data
python3 -m nanoowl.build_image_encoder_engine \
    data/owl_image_encoder_patch32.engine

# test the engine
cd examples
python3 owl_predict.py \
    --prompt="[an owl, a glove]" \
    --threshold=0.1 \
    --image_encoder_engine=../data/owl_image_encoder_patch32.engine

# install additional dependencies for realtime demo
pip install aiohttp

# run the realtime server
cd examples/tree_demo
python3 tree_demo.py ../../data/owl_image_encoder_patch32.engine
```

