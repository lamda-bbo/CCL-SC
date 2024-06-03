mkdir -p ./log

ARCH=resnet18
LOSS=csc
DATASET=celeba
PRETRAIN=1
MOM=0.9
seed=1
batch_size=64
epochs=50
m=0.999
k=300
rewards=0.5
t=0.1
save_model_step=25
  # Parsing arguments
  while getopts ":s:e:" flag; do
    case "${flag}" in
      s) seed=${OPTARG};;
      e) entropy=${OPTARG};;
      :)                                         # If expected argument omitted:
        echo "Error: -${OPTARG} requires an argument."
        exit_abnormal;;                          # Exit abnormally.
      *)                                         # If unknown (any other) option:
        exit_abnormal;;                          # Exit abnormally.
    esac
  done

  SAVE_DIR='./log/'${DATASET}_${ARCH}_${LOSS}_seed-${seed}

  ## train
  python -u train_CCL_SC.py --arch ${ARCH} --pretrain ${PRETRAIN} --sat-momentum ${MOM} --moco-m ${m} --moco-k ${k} --pretrain ${PRETRAIN}  --rewards ${rewards} --moco-t ${t}\
        --loss ${LOSS} --manualSeed ${seed} --train-batch ${batch_size} --save_model_step ${save_model_step}\
        --dataset ${DATASET} --save ${SAVE_DIR}  --epochs ${epochs}\
        2>&1 | tee -a ${SAVE_DIR}.log

  ### eval
  python -u train_moco_dev.py --arch ${ARCH} --manualSeed ${seed} --moco-m ${m} --moco-k ${k} --pretrain ${PRETRAIN}  --rewards ${rewards} --moco-t ${t}\
        --loss ${LOSS} --dataset ${DATASET} --train-batch ${batch_size} --epochs 619\
        --save ${SAVE_DIR} --evaluate \
        2>&1 | tee -a ${SAVE_DIR}.log