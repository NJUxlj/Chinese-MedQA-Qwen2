
current_file_dir=$(dirname "$0")
parent_dir=$(dirname "$current_file_dir")
LOG_DIR="$parent_dir/logs"
mkdir -p $LOG_DIR

DATE_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/base_trainer_$DATE_TIME.log"


nohup python $parent_dir/run_trainer/run_base_trainer.py \
    > $LOG_FILE 2>&1 &


echo "LOG FILE: ${LOG_FILE}"
