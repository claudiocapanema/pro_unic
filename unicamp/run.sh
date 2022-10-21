#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

# Download the CIFAR-10 dataset
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./dataset', download=True)"

# Download the EfficientNetB0 model
python -c "import torch; torch.hub.load( \
        'NVIDIA/DeepLearningExamples:torchhub', \
        'nvidia_efficientnet_b0', pretrained=True)"

for i in `seq 0 4`; do
  echo "Starting server $i"
  python server.py --algorithm=${i} --rounds=150&
  sleep 3  # Sleep for 3s to give the server enough time to start
  for j in `seq 0 9`; do
      echo "Starting client $j"
      python client.py --partition=${j} &
  done
  wait
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
