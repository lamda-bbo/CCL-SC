mkdir -p ./log

ARCH=resnet34
LOSS=csc
DATASET=imagenet
PRETRAIN=50
start_epochs=50
MOM=0.99
seed=1
m=0.999
k=10000
epochs=150
batch_size=256
rewards=0.1
t=0.1
  # Parsing arguments
  while getopts ":s:" flag; do
    case "${flag}" in
      s) seed=${OPTARG};;
      :)                                         # If expected argument omitted:
        echo "Error: -${OPTARG} requires an argument."
        exit_abnormal;;                          # Exit abnormally.
      *)                                         # If unknown (any other) option:
        exit_abnormal;;                          # Exit abnormally.
    esac
  done

  SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_seed-${seed}

  ### train
   python -u train_CCL_SC.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} --moco-m ${m} --rewards ${rewards} --moco-k ${k} --moco-t ${t}\
         --loss ${LOSS} --manualSeed ${seed}\
         --dataset ${DATASET} --save ${SAVE_DIR}   --epochs ${epochs} --train-batch ${batch_size} \
         --schedule 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 \
         2>&1 | tee -a ${SAVE_DIR}.log

  ### eval
  python -u train_moco_dev.py --arch ${ARCH} --manualSeed ${seed} --pretrain ${PRETRAIN} --sat-momentum ${MOM} --moco-m ${m} --rewards ${rewards} --moco-k ${k} --moco-t ${t}\
        --loss ${LOSS} --dataset ${DATASET} \
        --save ${SAVE_DIR} --evaluate   --epochs ${epochs} --train-batch ${batch_size}\
        --schedule 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 \
        2>&1 | tee -a ${SAVE_DIR}.log