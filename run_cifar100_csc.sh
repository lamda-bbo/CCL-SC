mkdir -p ./log
ARCH=vgg16_bn
LOSS=csc
DATASET=cifar100
PRETRAIN=150
MOM=0.9
seed=1
m=0.999
k=3000
rewards=1.0
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
  python -u train_CCL_SC.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} --moco-m ${m} --moco-k ${k}  --rewards ${rewards} --moco-t ${t}\
         --loss ${LOSS} --manualSeed ${seed}\
         --dataset ${DATASET} --save ${SAVE_DIR}  \
         2>&1 | tee -a ${SAVE_DIR}.log
  ### eval
  python -u train_moco_dev.py --arch ${ARCH} --manualSeed ${seed} --moco-m ${m} --moco-k ${k} --pretrain ${PRETRAIN}  --rewards ${rewards} --moco-t ${t}\
        --loss ${LOSS} --dataset ${DATASET}\
        --save ${SAVE_DIR} --evaluate \
        2>&1 | tee -a ${SAVE_DIR}.log